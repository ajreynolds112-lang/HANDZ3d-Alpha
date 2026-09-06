/**
 * Item inventory storage layer — Shards currency, owned items, and active
 * boost bookkeeping, persisted on the fighter save via localSaves.
 *
 * All rules live here so every caller gets the same enforcement:
 *  - obtainable limits (lifetime) per item
 *  - keepsakes are permanent one-time gifts
 *  - consumable rank-exclusivity on grant
 *  - "already active" vs synergy stacking on activation
 */
import type { CareerRosterState, Fighter, ItemInventory, SkillRefinement } from "@shared/schema";
import { DEFAULT_ITEM_INVENTORY, DEFAULT_SKILL_REFINEMENT } from "@shared/schema";
import * as localSaves from "@/lib/localSaves";
import { getItemDefinition, loadItemsConfig, type ItemDefinition, type ItemRarity } from "@/game/itemsConfig";
import { clampFamilyStacks, familyActiveStacks, getItemFamily } from "@/game/itemFamily";
import { countMap, countsDiffer, repairOwnedFromLedger } from "@/lib/itemLedger";
import { rollCrateDraw, rollItemOfRarity, type CrateId } from "@/game/cratesConfig";
import {
  AC_BOOST_MS, campSignature, effectOf, getLuckyCoinChance, isArmable, isInstantGrant,
  pruneExpiredBoosts, type ItemEffect, type ItemPerk,
} from "@/game/itemEffects";

/** Normalized inventory for any fighter, backfilling defaults for older saves. */
export function getInventory(fighter: Fighter | undefined | null): ItemInventory {
  const inv = (fighter?.itemInventory ?? null) as Partial<ItemInventory> | null;
  const rec = (v: unknown): Record<string, number> =>
    v && typeof v === "object" ? { ...(v as Record<string, number>) } : {};
  const ledger = repairOwnedFromLedger(
    rec(inv?.owned),
    rec(inv?.lifetimeObtained),
    inv?.itemsSpent && typeof inv.itemsSpent === "object" ? { ...inv.itemsSpent } : null,
  );
  return {
    owned: ledger.owned,
    itemsSpent: ledger.itemsSpent,
    lifetimeObtained: rec(inv?.lifetimeObtained),
    keepsakesReceived: Array.isArray(inv?.keepsakesReceived) ? [...inv.keepsakesReceived] : [],
    // Saves written before a family shared its synergy pool can hold more
    // stacks than the rule now allows — trim them on the way out and in.
    activeBoosts: clampFamilyStacks(rec(inv?.activeBoosts)),
    unseenItemIds: Array.isArray(inv?.unseenItemIds) ? [...inv.unseenItemIds] : [],
    boostWeek: rec(inv?.boostWeek),
    boostCamp: inv?.boostCamp && typeof inv.boostCamp === "object" ? { ...inv.boostCamp } : {},
    acBoost: inv?.acBoost && typeof inv.acBoost === "object" ? { ...inv.acBoost } : null,
    coachNotes: rec(inv?.coachNotes),
    recoveredFromLifetimeV1: inv?.recoveredFromLifetimeV1 === true ? true : undefined,
  };
}

/**
 * One-time locker recovery.
 *
 * `lifetimeObtained` records every copy of every item the save has ever been
 * given, so it can rebuild an inventory that was rolled back by a stale write:
 * every keepsake ever received goes back into `keepsakesReceived`, and every
 * non-consumable item is topped back up to the number of copies it was granted
 * (armed copies excluded — those legitimately left `owned`). Consumables are
 * left alone: they are meant to be used up.
 *
 * It runs once per save and records that on the save, so an item sold after the
 * recovery stays sold.
 */
export function recoverLockerFromLifetime(fighterId: string): void {
  const fighter = localSaves.getFighter(fighterId);
  if (!fighter) return;
  const inv = getInventory(fighter);
  if (inv.recoveredFromLifetimeV1) return;
  // Without the catalog nothing can be classified, so leave the save untouched
  // and let a later load do the recovery.
  if (loadItemsConfig().length === 0) return;

  // A save that predates the spend ledger has no real receipts — the ones
  // getInventory backfills are derived from the very counts this recovery
  // exists to rebuild, so those saves keep the original armed-only accounting.
  // Once a save carries its own receipts they are authoritative, and a copy
  // that was sold or used must stay gone.
  const stored = (fighter.itemInventory ?? null) as Partial<ItemInventory> | null;
  const hasLedger = !!stored?.itemsSpent && typeof stored.itemsSpent === "object";

  for (const [itemId, lifetimeRaw] of Object.entries(inv.lifetimeObtained)) {
    const lifetime = Math.max(0, Math.floor(lifetimeRaw ?? 0));
    const def = getItemDefinition(itemId);
    if (!def || lifetime <= 0) continue;
    if (def.isKeepsake && !inv.keepsakesReceived.includes(itemId)) inv.keepsakesReceived.push(itemId);
    if (def.isConsumable) continue;
    // Armed copies are counted in the ledger already, so they are only
    // subtracted separately on the legacy path.
    const gone = hasLedger ? (inv.itemsSpent?.[itemId] ?? 0) : (inv.activeBoosts[itemId] ?? 0);
    const want = Math.max(0, Math.min(lifetime, def.obtainableLimit ?? lifetime) - gone);
    if (want > (inv.owned[itemId] ?? 0)) inv.owned[itemId] = want;
    // The receipts backfilled for a pre-ledger save were guesses drawn from the
    // wiped counts — rewrite them to match what the save actually holds now, so
    // the ledger starts honest and a later loss is still repairable.
    if (!hasLedger) {
      const spent = Math.max(0, lifetime - (inv.owned[itemId] ?? 0));
      if (spent > 0) (inv.itemsSpent ?? (inv.itemsSpent = {}))[itemId] = spent;
      else delete inv.itemsSpent?.[itemId];
    }
  }
  inv.recoveredFromLifetimeV1 = true;
  localSaves.updateFighter(fighterId, { itemInventory: inv });
}

/**
 * Fold legacy holdings into what the keepsake rules expect: one owned copy,
 * nothing armed.
 *
 * A shipped item that used to be an activatable consumable can leave a save
 * holding several copies — which would stack a permanent effect that is meant
 * to be one-per-save — or holding none at all, because its only copy is sitting
 * in `activeBoosts` where a permanent effect is never read from. Both are
 * rewritten here, and the spend receipts are squared with the new count so the
 * ledger repair doesn't put the surplus straight back.
 *
 * Idempotent: once a keepsake sits at one owned copy with nothing armed there
 * is nothing to do, so this can run on every load.
 */
export function normalizeKeepsakeHoldings(fighterId: string): void {
  const fighter = localSaves.getFighter(fighterId);
  if (!fighter) return;
  const catalog = loadItemsConfig();
  if (catalog.length === 0) return;
  const inv = getInventory(fighter);
  let changed = false;

  for (const def of catalog) {
    if (!def.isKeepsake) continue;
    const owned = inv.owned[def.id] ?? 0;
    const lifetime = inv.lifetimeObtained[def.id] ?? 0;
    const armed = (inv.activeBoosts[def.id] ?? 0) > 0
      || inv.boostWeek?.[def.id] != null
      || inv.boostCamp?.[def.id] != null
      || inv.acBoost?.itemId === def.id;
    // Never seen by this save: leave it alone rather than gifting one.
    if (owned <= 0 && lifetime <= 0 && !armed && !inv.keepsakesReceived.includes(def.id)) continue;

    if (armed) {
      delete inv.activeBoosts[def.id];
      if (inv.boostWeek) delete inv.boostWeek[def.id];
      if (inv.boostCamp) delete inv.boostCamp[def.id];
      if (inv.acBoost?.itemId === def.id) inv.acBoost = null;
      changed = true;
    }
    const want = Math.min(Math.max(1, def.obtainableLimit), Math.max(1, owned));
    if (want !== owned) {
      inv.owned[def.id] = want;
      changed = true;
    }
    // The receipts have to agree with the count, or the next read's ledger
    // repair reads the gap as a bad write and restores the copies just removed.
    const spent = Math.max(0, lifetime - want);
    if (spent !== (inv.itemsSpent?.[def.id] ?? 0)) {
      if (spent > 0) (inv.itemsSpent ?? (inv.itemsSpent = {}))[def.id] = spent;
      else delete inv.itemsSpent?.[def.id];
      changed = true;
    }
    if (!inv.keepsakesReceived.includes(def.id)) {
      inv.keepsakesReceived.push(def.id);
      changed = true;
    }
  }
  if (!changed) return;
  // Shrinking is the point of the write, so it opts out of the upward merge —
  // safe because the inventory above was read from storage moments ago.
  localSaves.updateFighter(fighterId, { itemInventory: inv }, { allowInventoryShrink: true });
}

/**
 * Write a ledger repair back to the save.
 *
 * Reading an inventory already repairs it, but a read alone leaves the stored
 * save (and anything reading it raw) short until some later write happens to
 * persist the repaired object. Call this at load so a locker heals itself even
 * if the player never touches it that session.
 */
export function healLockerFromLedger(fighterId: string): void {
  const fighter = localSaves.getFighter(fighterId);
  if (!fighter) return;
  const stored = (fighter.itemInventory ?? null) as Partial<ItemInventory> | null;
  const inv = getInventory(fighter);
  const ownedChanged = countsDiffer(countMap(stored?.owned), inv.owned);
  const spentChanged = countsDiffer(countMap(stored?.itemsSpent), inv.itemsSpent ?? {});
  if (!ownedChanged && !spentChanged) return;
  localSaves.updateFighter(fighterId, { itemInventory: inv });
}

export function getShards(fighter: Fighter | undefined | null): number {
  return fighter?.shards ?? 0;
}

export interface ItemOpResult {
  ok: boolean;
  message?: string;
  fighter?: Fighter;
}

/**
 * Grant a copy of an item to the save. Enforces the lifetime obtainable limit
 * and consumable rank-exclusivity. Keepsakes are refused outright — they are
 * only ever won off an opponent wearing one (claimOpponentKeepsakes).
 */
export function grantItem(fighterId: string, itemId: string, playerRank?: number | null): ItemOpResult {
  const def = getItemDefinition(itemId);
  if (!def) return { ok: false, message: "Unknown item" };
  const fighter = localSaves.getFighter(fighterId);
  if (!fighter) return { ok: false, message: "Save not found" };
  const inv = getInventory(fighter);

  if (def.isKeepsake) {
    return { ok: false, message: `${def.name} can only be won off an opponent wearing it.` };
  }
  const lifetime = inv.lifetimeObtained[itemId] ?? 0;
  if (lifetime >= def.obtainableLimit) {
    return { ok: false, message: `${def.name} obtainable limit reached.` };
  }
  if (def.isConsumable && playerRank != null) {
    if (def.rankExclusiveLo != null && playerRank < def.rankExclusiveLo) {
      return { ok: false, message: `${def.name} is not obtainable at rank ${playerRank}.` };
    }
    if (def.rankExclusiveHi != null && playerRank > def.rankExclusiveHi) {
      return { ok: false, message: `${def.name} is not obtainable at rank ${playerRank}.` };
    }
  }

  inv.owned[itemId] = (inv.owned[itemId] ?? 0) + 1;
  inv.lifetimeObtained[itemId] = lifetime + 1;
  markUnseen(inv, itemId);
  const dupes = applyLuckyCoin(fighter, inv, [def], Math.random);
  const updated = localSaves.updateFighter(fighterId, { itemInventory: inv });
  return {
    ok: true,
    message: dupes.length > 0 ? `Lucky Coin doubled your ${def.name}!` : undefined,
    fighter: updated,
  };
}

/** Flag a freshly obtained item as unseen so the locker shows its notification dot. */
function markUnseen(inv: ItemInventory, itemId: string): void {
  const unseen = inv.unseenItemIds ?? [];
  if (!unseen.includes(itemId)) unseen.push(itemId);
  inv.unseenItemIds = unseen;
}

export interface CrateRollResult {
  /** Item ids drawn, in draw order (may contain duplicates). Empty when nothing was eligible. */
  itemIds: string[];
  /**
   * Inventory to persist. Null when nothing was drawn. The caller writes this —
   * fold it into whatever save write it is already making so the crate and the
   * result it came from land in a single storage update.
   */
  inventory: ItemInventory | null;
  /** Subset of `itemIds` that Lucky Coin duplicated. */
  luckyCoinItemIds?: string[];
}

/**
 * Roll a crate for a fighter and build the resulting inventory *without*
 * writing it.
 *
 * The roll and the inventory it produces come from one catalog snapshot and one
 * read of the fighter, so the ids returned are exactly the ids the returned
 * inventory contains — the reveal can never show an item the locker didn't get.
 * Lucky Coin duplicates ride along in the same snapshot instead of appearing
 * after the reveal.
 * Draw eligibility (0% rarity, lifetime caps, keepsake one-time rule, consumable
 * rank exclusivity) is enforced by the roll itself.
 *
 * `allowKeepsakes` opens the draw to keepsakes for callers that are allowed to
 * hand them out (the gym). A drawn keepsake is recorded in `keepsakesReceived`
 * here, exactly as winning one off an opponent would.
 */
export function rollCrateForFighter(
  fighter: Fighter | undefined | null,
  crateId: CrateId,
  playerRank?: number | null,
  allowKeepsakes = false,
): CrateRollResult {
  if (!fighter) return { itemIds: [], inventory: null };
  const catalog = loadItemsConfig();
  const inv = getInventory(fighter);
  const drawn = rollCrateDraw(crateId, {
    inventory: inv, playerRank, items: catalog, allowKeepsakes,
    refinementMaxed: isRefinementMaxed(fighter),
  });
  if (drawn.length === 0) return { itemIds: [], inventory: null };
  for (const def of drawn) {
    inv.owned[def.id] = (inv.owned[def.id] ?? 0) + 1;
    inv.lifetimeObtained[def.id] = (inv.lifetimeObtained[def.id] ?? 0) + 1;
    if (def.isKeepsake && !inv.keepsakesReceived.includes(def.id)) inv.keepsakesReceived.push(def.id);
    markUnseen(inv, def.id);
  }
  const dupes = applyLuckyCoin(fighter, inv, drawn, Math.random);
  return { itemIds: [...drawn.map(d => d.id), ...dupes], inventory: inv, luckyCoinItemIds: dupes };
}

/**
 * Grant one random item of a given rarity — the sparring-win reward. Same
 * contract as rollCrateForFighter: one catalog snapshot, one read of the
 * fighter, and an inventory the caller folds into its own save write.
 *
 * `maxRarity` caps the draw including the exhausted-tier fallback, for callers
 * whose reward has an advertised ceiling. `allowKeepsakes` works exactly as it
 * does for a crate: opt-in, one of each ever, recorded in `keepsakesReceived`.
 */
export function rollRarityItemForFighter(
  fighter: Fighter | undefined | null,
  rarity: ItemRarity,
  playerRank?: number | null,
  maxRarity?: ItemRarity | null,
  allowKeepsakes = false,
): CrateRollResult {
  if (!fighter) return { itemIds: [], inventory: null };
  const catalog = loadItemsConfig();
  const inv = getInventory(fighter);
  const drawn = rollItemOfRarity(rarity, {
    inventory: inv, playerRank, items: catalog, maxRarity, allowKeepsakes,
    refinementMaxed: isRefinementMaxed(fighter),
  });
  if (!drawn) return { itemIds: [], inventory: null };
  inv.owned[drawn.id] = (inv.owned[drawn.id] ?? 0) + 1;
  inv.lifetimeObtained[drawn.id] = (inv.lifetimeObtained[drawn.id] ?? 0) + 1;
  if (drawn.isKeepsake && !inv.keepsakesReceived.includes(drawn.id)) inv.keepsakesReceived.push(drawn.id);
  markUnseen(inv, drawn.id);
  const dupes = applyLuckyCoin(fighter, inv, [drawn], Math.random);
  return { itemIds: [drawn.id, ...dupes], inventory: inv, luckyCoinItemIds: dupes };
}

/**
 * Does this save already hold that keepsake?
 *
 * The three records can disagree — a save recovered from a stale write, an
 * imported one, or a locker written before `keepsakesReceived` existed can
 * hold a keepsake it never recorded as received. Every keepsake rule reads
 * this one predicate rather than trusting the received list on its own, so
 * "the player has it" means the same thing to the claim (which refuses a
 * duplicate) and to the opponent roll (which must not carry an unwinnable
 * one).
 */
function holdsKeepsake(inv: ItemInventory, def: ItemDefinition): boolean {
  return inv.keepsakesReceived.includes(def.id)
    || (inv.owned[def.id] ?? 0) > 0
    || (inv.lifetimeObtained[def.id] ?? 0) >= def.obtainableLimit;
}

/**
 * Every keepsake this save already holds. Career opponents are handed one the
 * player is still missing, so this is the list the roll excludes.
 */
export function heldKeepsakeIds(fighter: Fighter | undefined | null): string[] {
  if (!fighter) return [];
  const inv = getInventory(fighter);
  return loadItemsConfig().filter(d => d.isKeepsake && holdsKeepsake(inv, d)).map(d => d.id);
}

/**
 * Beating a career opponent hands over every keepsake in their kit that the
 * player doesn't already hold. Keepsakes are one-of-a-kind, so one already in
 * the locker is skipped rather than duplicated, and Lucky Coin never applies.
 * Same contract as rollCrateForFighter: one catalog snapshot, one read of the
 * fighter, and an inventory the caller folds into its own save write.
 */
export function claimOpponentKeepsakes(
  fighter: Fighter | undefined | null,
  opponentItems: Record<string, number> | undefined | null,
): CrateRollResult {
  if (!fighter || !opponentItems) return { itemIds: [], inventory: null };
  const catalog = loadItemsConfig();
  const inv = getInventory(fighter);
  const claimed: string[] = [];
  for (const [itemId, stacks] of Object.entries(opponentItems)) {
    if ((stacks ?? 0) <= 0) continue;
    const def = catalog.find(d => d.id === itemId);
    if (!def?.isKeepsake) continue;
    if (holdsKeepsake(inv, def)) continue;
    inv.owned[itemId] = (inv.owned[itemId] ?? 0) + 1;
    inv.lifetimeObtained[itemId] = (inv.lifetimeObtained[itemId] ?? 0) + 1;
    inv.keepsakesReceived.push(itemId);
    markUnseen(inv, itemId);
    claimed.push(itemId);
  }
  if (claimed.length === 0) return { itemIds: [], inventory: null };
  return { itemIds: claimed, inventory: inv };
}

/** Lucky Coin's id — the collection's last box to tick before keepsakes open up. */
const LUCKY_COIN_ID = "lucky_coin";

/**
 * True once the player has received every keepsake in the catalog and every
 * copy of Lucky Coin. Career opponents stop being restricted to keepsakes the
 * player is still missing at that point — there is nothing left to win, so
 * they can wear any of them (see rollOpponentItems).
 */
export function hasAllKeepsakesAndCoins(fighter: Fighter | undefined | null): boolean {
  if (!fighter) return false;
  const catalog = loadItemsConfig();
  const keepsakes = catalog.filter(d => d.isKeepsake);
  if (keepsakes.length === 0) return false;
  const inv = getInventory(fighter);
  if (!keepsakes.every(d => holdsKeepsake(inv, d))) return false;
  const coin = catalog.find(d => d.id === LUCKY_COIN_ID);
  if (!coin) return true;
  const limit = Math.max(1, coin.obtainableLimit ?? 1);
  return Math.max(inv.owned[LUCKY_COIN_ID] ?? 0, inv.lifetimeObtained[LUCKY_COIN_ID] ?? 0) >= limit;
}

/** True while the save holds items the player hasn't looked at in the Locker. */
export function hasUnseenItems(fighter: Fighter | undefined | null): boolean {
  return (getInventory(fighter).unseenItemIds ?? []).length > 0;
}

/** Clear the unseen-item flag — called when the Locker is opened. */
export function markItemsSeen(fighterId: string): Fighter | undefined {
  const fighter = localSaves.getFighter(fighterId);
  if (!fighter) return undefined;
  const inv = getInventory(fighter);
  if ((inv.unseenItemIds ?? []).length === 0) return fighter;
  inv.unseenItemIds = [];
  return localSaves.updateFighter(fighterId, { itemInventory: inv });
}

/** Sell owned copies of an item for Shards / Force / Diamonds. */
export function sellItem(fighterId: string, itemId: string, qty = 1): ItemOpResult {
  const def = getItemDefinition(itemId);
  if (!def) return { ok: false, message: "Unknown item" };
  if (!def.sellable || def.isKeepsake) return { ok: false, message: `${def.name} can't be sold.` };
  const fighter = localSaves.getFighter(fighterId);
  if (!fighter) return { ok: false, message: "Save not found" };
  const inv = getInventory(fighter);
  const owned = inv.owned[itemId] ?? 0;
  const sellQty = Math.min(Math.max(1, Math.floor(qty)), owned);
  if (sellQty <= 0) return { ok: false, message: `No ${def.name} to sell.` };

  // Armed copies have already left `owned`, so selling never touches them.
  inv.owned[itemId] = owned - sellQty;
  if (inv.owned[itemId] <= 0) delete inv.owned[itemId];
  recordSpent(inv, itemId, sellQty);
  const gainedShards = sellQty * Math.max(0, def.sellShards);
  const gainedForce = sellQty * Math.max(0, def.sellForce);
  const gainedDiamonds = sellQty * Math.max(0, def.sellDiamonds);
  const updated = localSaves.updateFighter(fighterId, {
    itemInventory: inv,
    shards: (fighter.shards ?? 0) + gainedShards,
    force: (fighter.force ?? 0) + gainedForce,
    diamonds: (fighter.diamonds ?? 0) + gainedDiamonds,
  }, { allowInventoryShrink: true });
  const parts: string[] = [];
  if (gainedShards > 0) parts.push(`${gainedShards.toLocaleString()} Shards`);
  if (gainedForce > 0) parts.push(`${gainedForce.toLocaleString()} Force`);
  if (gainedDiamonds > 0) parts.push(`${gainedDiamonds.toLocaleString()} Diamonds`);
  const payout = parts.length > 0 ? parts.join(" + ") : "nothing";
  return { ok: true, message: `Sold ${sellQty}× ${def.name} for ${payout}`, fighter: updated };
}

// ---------------------------------------------------------------------------
// Using items
// ---------------------------------------------------------------------------

/** Refinement categories Sparta's Trophy can max out. */
export const REFINEMENT_CATEGORIES = [
  "pressureFighter", "precisionStriker", "jabPower", "hookPower", "uppercutPower", "bruiser", "koArtist",
  "ironChin", "slippery", "guardMaster", "duckRecovery", "punchRolling",
  "fastTwitch", "heartRefinement", "chinHitter", "technician", "lifeDrain",
] as const;
export type RefinementCategory = typeof REFINEMENT_CATEGORIES[number];
export const REFINEMENT_MAX = 100;

/**
 * Is every refinement category already at its ceiling?
 *
 * Read straight off the fighter's saved refinement, so the answer is whatever
 * the career itself carries — an exported save that comes back in is still
 * maxed, and a save that isn't stays eligible. No separate flag to keep in
 * sync, nothing browser-level to lose on an import.
 */
export function isRefinementMaxed(fighter: Fighter | undefined | null): boolean {
  const ref = (fighter?.skillRefinement ?? null) as Partial<SkillRefinement> | null;
  if (!ref) return false;
  return REFINEMENT_CATEGORIES.every(cat => ((ref[cat] as number | undefined) ?? 0) >= REFINEMENT_MAX);
}

/** Which track a refinement category belongs to — maxing it unlocks that track. */
const CATEGORY_TRACK: Record<RefinementCategory, keyof Pick<SkillRefinement, "offenseUnlocked" | "defenseUnlocked" | "fightIqUnlocked">> = {
  pressureFighter: "offenseUnlocked", precisionStriker: "offenseUnlocked", jabPower: "offenseUnlocked",
  hookPower: "offenseUnlocked", uppercutPower: "offenseUnlocked", bruiser: "offenseUnlocked",
  koArtist: "offenseUnlocked",
  ironChin: "defenseUnlocked", slippery: "defenseUnlocked", guardMaster: "defenseUnlocked",
  duckRecovery: "defenseUnlocked", punchRolling: "defenseUnlocked",
  fastTwitch: "fightIqUnlocked", heartRefinement: "fightIqUnlocked",
  chinHitter: "fightIqUnlocked", technician: "fightIqUnlocked", lifeDrain: "fightIqUnlocked",
};

export interface UseItemOptions {
  /** Current roster state — needed to scope week/camp boosts. */
  roster?: CareerRosterState | null;
  /** Sparta's Trophy: which refinement category to max out. */
  refinementCategory?: RefinementCategory;
  /** AC boosts: the player already confirmed replacing the live one. */
  confirmReplaceAc?: boolean;
  nowMs?: number;
}

/**
 * Use one copy of an item.
 *
 * Instant grants pay out immediately; everything else is *armed* — the copy
 * leaves `owned` and shows up in `activeBoosts` (or the single AC slot) until
 * its window closes. Permanent items are never used: owning them is enough.
 */
export function useItem(fighterId: string, itemId: string, opts: UseItemOptions = {}): ItemOpResult {
  const def = getItemDefinition(itemId);
  if (!def) return { ok: false, message: "Unknown item" };
  const effect = effectOf(def);
  if (!effect) return { ok: false, message: `${def.name} has no effect yet.` };
  if (effect.duration === "permanent") {
    return { ok: false, message: `${def.name} works automatically while you own it.` };
  }
  const fighter = localSaves.getFighter(fighterId);
  if (!fighter) return { ok: false, message: "Save not found" };
  const nowMs = opts.nowMs ?? Date.now();
  const inv = getInventory(fighter);
  pruneExpiredBoosts(inv, opts.roster, nowMs);

  if ((inv.owned[itemId] ?? 0) <= 0) return { ok: false, message: `No ${def.name} left.` };

  if (isInstantGrant(def)) return applyInstantGrant(fighterId, fighter, inv, def, effect, opts);
  return armBoost(fighterId, fighter, inv, def, effect, opts, nowMs);
}

/**
 * Take one copy out of the locker and write the receipt for it. Every copy that
 * leaves without one is treated as lost and restored on the next read, so this
 * (and sellItem) must stay the only ways an owned count ever goes down.
 */
function consumeOne(inv: ItemInventory, itemId: string): void {
  const owned = (inv.owned[itemId] ?? 0) - 1;
  if (owned <= 0) delete inv.owned[itemId];
  else inv.owned[itemId] = owned;
  recordSpent(inv, itemId, 1);
}

/** Book copies as deliberately spent (sold, used or armed). */
function recordSpent(inv: ItemInventory, itemId: string, qty: number): void {
  if (qty <= 0) return;
  const spent = inv.itemsSpent ?? (inv.itemsSpent = {});
  spent[itemId] = Math.max(0, Math.floor(spent[itemId] ?? 0)) + qty;
}

function applyInstantGrant(
  fighterId: string,
  fighter: Fighter,
  inv: ItemInventory,
  def: ItemDefinition,
  effect: ItemEffect,
  opts: UseItemOptions,
): ItemOpResult {
  const patch: Partial<Fighter> = {};
  const parts: string[] = [];

  if (effect.grantForce) {
    patch.force = (fighter.force ?? 0) + effect.grantForce;
    parts.push(`${effect.grantForce.toLocaleString()} Force`);
  }
  if (effect.grantDiamonds) {
    patch.diamonds = (fighter.diamonds ?? 0) + effect.grantDiamonds;
    parts.push(`${effect.grantDiamonds.toLocaleString()} Diamonds`);
  }
  if (effect.grantRefinementPoints || effect.maxRefinementCategory) {
    // Spread the stored refinement so permanent fields (costReduction) survive.
    const ref: SkillRefinement = { ...DEFAULT_SKILL_REFINEMENT, ...(fighter.skillRefinement ?? {}) };
    if (effect.grantRefinementPoints) {
      ref.availablePoints = (ref.availablePoints ?? 0) + effect.grantRefinementPoints;
      parts.push(`${effect.grantRefinementPoints} refinement point${effect.grantRefinementPoints === 1 ? "" : "s"}`);
    }
    if (effect.maxRefinementCategory) {
      const cat = opts.refinementCategory;
      if (!cat || !REFINEMENT_CATEGORIES.includes(cat)) {
        return { ok: false, message: "Choose a refinement category to max out." };
      }
      if ((ref[cat] ?? 0) >= REFINEMENT_MAX) {
        return { ok: false, message: "That refinement category is already maxed." };
      }
      ref[cat] = REFINEMENT_MAX;
      ref[CATEGORY_TRACK[cat]] = true;
      parts.push(`${cat} maxed out`);
    }
    patch.skillRefinement = ref;
  }

  consumeOne(inv, def.id);
  patch.itemInventory = inv;
  const updated = localSaves.updateFighter(fighterId, patch, { allowInventoryShrink: true });
  return { ok: true, message: `${def.name} used — gained ${parts.join(", ") || "nothing"}.`, fighter: updated };
}

function armBoost(
  fighterId: string,
  fighter: Fighter,
  inv: ItemInventory,
  def: ItemDefinition,
  effect: ItemEffect,
  opts: UseItemOptions,
  nowMs: number,
): ItemOpResult {
  // AC boosts share one slot: arming a new one replaces the live one outright
  // (no synergy between AC boosts) and restarts the 24h timer.
  if (effect.duration === "24h") {
    const live = inv.acBoost && inv.acBoost.expiresAtMs > nowMs ? inv.acBoost : null;
    if (live && !opts.confirmReplaceAc) {
      const liveDef = getItemDefinition(live.itemId);
      return {
        ok: false,
        message: `REPLACE_AC:${liveDef?.name ?? live.itemId}`,
      };
    }
    consumeOne(inv, def.id);
    inv.acBoost = { itemId: def.id, expiresAtMs: nowMs + AC_BOOST_MS };
    const updated = localSaves.updateFighter(fighterId, { itemInventory: inv }, { allowInventoryShrink: true });
    return { ok: true, message: `${def.name} active for the next 24 hours.`, fighter: updated };
  }

  if (effect.duration === "camp" && campSignature(opts.roster) == null) {
    return { ok: false, message: `${def.name} can only be used during a fight camp.` };
  }

  // Synergy pools across the whole tier ladder: three Electrolyte Bottles means
  // three in total, whether they're all tier I or one each of I, III and VI.
  // Without synergy nothing changes — each tier keeps its own single slot.
  const active = inv.activeBoosts[def.id] ?? 0;
  const family = getItemFamily(def);
  if (!family.synergy) {
    if (active >= 1) return { ok: false, message: `${def.name} already active!` };
  } else {
    const maxStacks = Math.max(1, family.stackCount);
    const famActive = familyActiveStacks(inv.activeBoosts, family.memberIds);
    if (famActive >= maxStacks) {
      const label = family.tiered ? family.baseName : def.name;
      return { ok: false, message: `${label} is at its maximum of ${maxStacks} stacks.` };
    }
  }

  consumeOne(inv, def.id);
  inv.activeBoosts[def.id] = active + 1;
  if (effect.duration === "week") {
    inv.boostWeek = { ...(inv.boostWeek ?? {}), [def.id]: opts.roster?.weekNumber ?? 0 };
  }
  if (effect.duration === "camp") {
    inv.boostCamp = { ...(inv.boostCamp ?? {}), [def.id]: campSignature(opts.roster)! };
  }
  const updated = localSaves.updateFighter(fighterId, { itemInventory: inv }, { allowInventoryShrink: true });
  return { ok: true, message: `${def.name} activated.`, fighter: updated };
}

/** Can this item be used from the Locker right now? */
export function isUsable(def: ItemDefinition | undefined | null): boolean {
  return isInstantGrant(def) || isArmable(def);
}

// ---------------------------------------------------------------------------
// Consuming armed boosts
// ---------------------------------------------------------------------------

export type BoostTrigger = "fight" | "weightLifting" | "heavyBag" | "sparring";

const TRIGGER_DURATION: Record<BoostTrigger, string> = {
  fight: "nextFight",
  weightLifting: "nextWL",
  heavyBag: "nextHB",
  sparring: "nextSpar",
};

/**
 * Burn the one-shot boosts tied to an event that just resolved, and drop any
 * week/camp/24h boost whose window has closed.
 *
 * Returns the inventory to persist, or null when nothing changed — callers fold
 * it into the save write they are already making.
 */
/**
 * Return the fighter with its **persisted** item inventory attached.
 *
 * The Locker arms and sells items by writing straight to the save, so any
 * in-memory fighter snapshot taken before the player opened it carries a stale
 * inventory. Reading a boost off that snapshot silently drops it; *writing* one
 * back erases whatever the Locker just armed. Every boost read and every boost
 * consumption must go through here first.
 *
 * Only the inventory is refreshed — the rest of the snapshot may legitimately
 * be newer than the save (mid-fight level-ups, unsaved allocations).
 */
export function withSavedInventory<T extends Fighter>(fighter: T): T;
export function withSavedInventory<T extends Fighter>(fighter: T | null | undefined): T | null | undefined;
export function withSavedInventory<T extends Fighter>(fighter: T | null | undefined): T | null | undefined {
  if (!fighter) return fighter;
  const saved = localSaves.getFighter(fighter.id);
  if (!saved || saved.itemInventory === fighter.itemInventory) return fighter;
  return { ...fighter, itemInventory: saved.itemInventory };
}

export function consumeBoostsFor(
  fighter: Fighter | undefined | null,
  // A session can answer to more than one trigger: a Doghouse round is both a
  // real bout and a sparring session, so it burns what each of them arms.
  trigger: BoostTrigger | readonly BoostTrigger[],
  roster?: CareerRosterState | null,
  opts: { sparring?: boolean; nowMs?: number; skipPerks?: readonly ItemPerk[] } = {},
): ItemInventory | null {
  if (!fighter) return null;
  const nowMs = opts.nowMs ?? Date.now();
  const inv = getInventory(fighter);
  let changed = pruneExpiredBoosts(inv, roster, nowMs);
  const triggers: readonly BoostTrigger[] = Array.isArray(trigger) ? trigger : [trigger as BoostTrigger];
  const want = triggers.map(t => TRIGGER_DURATION[t]);
  for (const itemId of Object.keys(inv.activeBoosts)) {
    const effect = effectOf(getItemDefinition(itemId));
    if (!effect || !want.includes(effect.duration)) continue;
    // A bout only burns the boosts it could actually use. Sparring-flag modes
    // (sparring, Nightmare, Doghouse) never pay career rewards, so they must
    // leave career-only boosts armed for the real career bout.
    if (opts.sparring && effect.fightScope === "career") continue;
    // A perk the session never actually delivered stays armed — the player
    // must not lose a ticket to a session that ignored it.
    if (effect.perk && opts.skipPerks?.includes(effect.perk)) continue;
    delete inv.activeBoosts[itemId];
    changed = true;
  }
  return changed ? inv : null;
}

/**
 * Drop lapsed week/camp/24h boosts. Call when the roster week advances or a
 * camp ends so the HUD and the Locker stop showing dead boosts.
 */
export function pruneBoosts(
  fighter: Fighter | undefined | null,
  roster?: CareerRosterState | null,
  nowMs = Date.now(),
): ItemInventory | null {
  if (!fighter) return null;
  const inv = getInventory(fighter);
  return pruneExpiredBoosts(inv, roster, nowMs) ? inv : null;
}

/**
 * Wipe every Training Frenzy-style boost the player wasted by simulating the
 * week instead of training through it.
 */
export function nullifySimulatedWeekBoosts(fighter: Fighter | undefined | null): ItemInventory | null {
  if (!fighter) return null;
  const inv = getInventory(fighter);
  let changed = false;
  for (const itemId of Object.keys(inv.activeBoosts)) {
    if (!effectOf(getItemDefinition(itemId))?.nullifiedBySimulateWeek) continue;
    delete inv.activeBoosts[itemId];
    if (inv.boostWeek) delete inv.boostWeek[itemId];
    changed = true;
  }
  return changed ? inv : null;
}

/**
 * Coach's Notes: bank a permanent percentage on a random stat after a camp
 * sparring session. Returns the inventory to persist, or null when inactive.
 */
export function accrueCoachNotes(
  fighter: Fighter | undefined | null,
  rate: number,
  stats: readonly string[],
  rng: () => number = Math.random,
): { inventory: ItemInventory; stat: string; pct: number } | null {
  if (!fighter || rate <= 0 || stats.length === 0) return null;
  const inv = getInventory(fighter);
  const stat = stats[Math.floor(rng() * stats.length)] ?? stats[0];
  const notes = inv.coachNotes ?? {};
  const next = Math.min(100, (notes[stat] ?? 0) + rate);
  if (next === (notes[stat] ?? 0)) return null;
  notes[stat] = next;
  inv.coachNotes = notes;
  return { inventory: inv, stat, pct: next };
}

// ---------------------------------------------------------------------------
// Lucky Coin
// ---------------------------------------------------------------------------

/**
 * Roll Lucky Coin against each drawn item and append the duplicates it wins.
 * Duplicates still respect lifetime limits, so a maxed item can't double.
 */
function applyLuckyCoin(
  fighter: Fighter,
  inv: ItemInventory,
  drawn: ItemDefinition[],
  rng: () => number,
): string[] {
  const chance = getLuckyCoinChance(fighter);
  if (chance <= 0) return [];
  const dupes: string[] = [];
  for (const def of drawn) {
    if (rng() >= chance) continue;
    if (def.isKeepsake) continue; // keepsakes are one-time by definition
    if ((inv.lifetimeObtained[def.id] ?? 0) >= def.obtainableLimit) continue;
    inv.owned[def.id] = (inv.owned[def.id] ?? 0) + 1;
    inv.lifetimeObtained[def.id] = (inv.lifetimeObtained[def.id] ?? 0) + 1;
    markUnseen(inv, def.id);
    dupes.push(def.id);
  }
  return dupes;
}

export type { ItemDefinition, ItemInventory };
export { DEFAULT_ITEM_INVENTORY };
