/**
 * Items config — the admin-defined catalog of every item in the game.
 *
 * Items are created/tuned in the Neural Network → Items editor and persisted
 * to localStorage (same pattern as rosterGenConfig). The shipped catalog in
 * `defaultItems.ts` is always present and saved entries override it by id;
 * older saved configs are backfilled field-by-field so new fields never crash.
 */
import { DEFAULT_ITEMS } from "@/game/defaultItems";

export type ItemRarity =
  | "Journeyman"
  | "Contender"
  | "Elite"
  | "Champion"
  | "Undisputed"
  | "GOAT";

export const ITEM_RARITIES: ItemRarity[] = [
  "Journeyman", "Contender", "Elite", "Champion", "Undisputed", "GOAT",
];

export const RARITY_COLORS: Record<ItemRarity, string> = {
  Journeyman: "#22aa44",
  Contender: "#3b82f6",
  Elite: "#a855f7",
  Champion: "#f5b301",
  Undisputed: "#ffffff",
  /**
   * GOAT is drawn as an animated starfield rather than a flat colour — this is
   * only the fallback for places that can't take a CSS class (canvas, title
   * attributes, inline SVG fills).
   */
  GOAT: "#8b5cf6",
};

/**
 * Rarity styling helpers.
 *
 * Every tier below GOAT is a flat colour applied inline. GOAT gets a
 * pulsating black/purple starfield defined in `index.css`, which can only be
 * expressed as a class — so each call site pairs the style with the class and
 * lets the class win for GOAT.
 */
export function rarityTextStyle(rarity: ItemRarity): { color?: string } {
  return rarity === "GOAT" ? {} : { color: RARITY_COLORS[rarity] };
}

export function rarityTextClass(rarity: ItemRarity): string {
  return rarity === "GOAT" ? "rarity-goat-text" : "";
}

export function rarityBorderStyle(rarity: ItemRarity): { borderColor?: string } {
  return rarity === "GOAT" ? {} : { borderColor: RARITY_COLORS[rarity] };
}

export function rarityBorderClass(rarity: ItemRarity): string {
  return rarity === "GOAT" ? "rarity-goat-border" : "";
}

export interface ItemDefinition {
  id: string;
  name: string;
  /** Emoji / short text drawn as the item's icon in the Locker grid. */
  icon: string;
  /** Optional isometric pixel-art PNG (public path). Falls back to `icon` when absent. */
  iconImage?: string;
  /**
   * Key into the code-defined effect table (`itemEffects.ts`). Built-in items
   * use their own id; admin-authored items can point at any shipped effect.
   */
  effectId?: string;
  rarity: ItemRarity;
  /** Category flags — an item can be a Boost Item and also Consumable/Keepsake. */
  isBoost: boolean;
  isConsumable: boolean;
  isKeepsake: boolean;
  /** Max copies a save can ever obtain. Consumables lock to 9999; keepsakes default 1 (editable). */
  obtainableLimit: number;
  /** Synergy items may stack the same active boost multiple times. */
  synergy: boolean;
  /** Max simultaneous stacks when synergy is on (1–9999). */
  synergyStackCount: number;
  /**
   * How duplicate stacks combine. False (the default) compounds — each stack
   * multiplies the running total, so n copies of a 5× item give 5^n×. True
   * makes them additive: n copies give 5n×. Only multiplicative effects are
   * affected; percentage and flat effects always added linearly.
   */
  synergyAdditive: boolean;
  /** When false the item can't be sold at all; the sell amounts below are ignored. */
  sellable: boolean;
  /** Sell payout in Shards (0–999,999,999). */
  sellShards: number;
  /** Sell payout in Force (0–999,999,999). */
  sellForce: number;
  /** Sell payout in Diamonds (0–999,999,999). */
  sellDiamonds: number;
  /** Drop weighting used by reward draws, 0–100%. */
  rarityPercent: number;
  /** Consumable rank-exclusivity — only obtainable while player rank is within [lo, hi]. Null = any rank. */
  rankExclusiveLo: number | null;
  rankExclusiveHi: number | null;
}

const STORAGE_KEY = "handz_items_config";

export const MAX_SYNERGY_STACK = 9999;
export const CONSUMABLE_LIMIT = 9999;
/** Cap on every per-currency sell amount. */
export const MAX_SELL_AMOUNT = 999999999;

function genItemId(): string {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function buildNewItem(): ItemDefinition {
  return {
    id: genItemId(),
    name: "New Item",
    icon: "🎁",
    rarity: "Journeyman",
    isBoost: false,
    isConsumable: false,
    isKeepsake: false,
    obtainableLimit: 1,
    synergy: false,
    synergyStackCount: 1,
    synergyAdditive: false,
    sellable: false,
    sellShards: 0,
    sellForce: 0,
    sellDiamonds: 0,
    rarityPercent: 0,
    rankExclusiveLo: null,
    rankExclusiveHi: null,
  };
}

/** Backfill/normalize a raw stored item so configs saved before new fields existed keep working. */
function normalizeItem(raw: unknown): ItemDefinition | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.id !== "string" || r.id === "") return null;
  const num = (v: unknown, def: number) =>
    typeof v === "number" && Number.isFinite(v) ? v : def;
  const rarity = ITEM_RARITIES.includes(r.rarity as ItemRarity)
    ? (r.rarity as ItemRarity)
    : "Journeyman";
  const isConsumable = r.isConsumable === true;
  const isKeepsake = !isConsumable && r.isKeepsake === true;
  const money = (v: unknown, def: number) =>
    Math.max(0, Math.min(MAX_SELL_AMOUNT, Math.round(num(v, def))));
  // Legacy configs stored a single Shards "sellPrice" and every item was sellable.
  const legacyShards = Math.max(0, Math.round(num(r.sellPrice, 0)));
  // Keepsakes are never sellable, whatever an older config or hand-edited import claims.
  const sellable = isKeepsake
    ? false
    : typeof r.sellable === "boolean" ? r.sellable : legacyShards > 0;
  const item: ItemDefinition = {
    id: r.id,
    name: typeof r.name === "string" ? r.name : "Unnamed Item",
    icon: typeof r.icon === "string" && r.icon !== "" ? r.icon : "🎁",
    iconImage: typeof r.iconImage === "string" && r.iconImage !== "" ? r.iconImage : undefined,
    effectId: typeof r.effectId === "string" && r.effectId !== "" ? r.effectId : undefined,
    rarity,
    isBoost: r.isBoost === true,
    isConsumable,
    isKeepsake,
    obtainableLimit: Math.max(1, Math.round(num(r.obtainableLimit, 1))),
    synergy: r.synergy === true,
    synergyStackCount: Math.max(1, Math.min(MAX_SYNERGY_STACK, Math.round(num(r.synergyStackCount, 1)))),
    synergyAdditive: r.synergyAdditive === true,
    sellable,
    sellShards: money(r.sellShards, legacyShards),
    sellForce: money(r.sellForce, 0),
    sellDiamonds: money(r.sellDiamonds, 0),
    rarityPercent: Math.max(0, Math.min(100, num(r.rarityPercent, 0))),
    rankExclusiveLo: typeof r.rankExclusiveLo === "number" && Number.isFinite(r.rankExclusiveLo) ? r.rankExclusiveLo : null,
    rankExclusiveHi: typeof r.rankExclusiveHi === "number" && Number.isFinite(r.rankExclusiveHi) ? r.rankExclusiveHi : null,
  };
  if (item.isConsumable) item.obtainableLimit = CONSUMABLE_LIMIT;
  return item;
}

// Cache keyed on the raw string so hot paths don't re-parse but saves apply immediately.
let cachedRaw: string | null = null;
let cachedItems: ItemDefinition[] | null = null;

/**
 * The shipped catalog plus whatever the admin saved.
 *
 * Built-ins always exist so a fresh save has the full item list; a saved entry
 * with the same id overrides its built-in field-for-field (that's how the Items
 * editor tunes shipped items), and admin-authored items are appended.
 */
/**
 * True when the saved copy still points at a family's single untinted sprite
 * while the shipped item has moved to its own rarity-tinted variant of that
 * exact file (see script/recolorItemIcons.ts). Configs saved before the tinted
 * art existed carry the old path, which would otherwise override the new art
 * and show every tier of Headgear Strap / Refinement Tome in the same colour.
 * Genuinely custom artwork points somewhere else and is left alone.
 */
function isPreTintArt(builtInImage: string | undefined, savedImage: string | undefined, rarity: ItemRarity): boolean {
  if (!builtInImage || !savedImage || builtInImage === savedImage) return false;
  const suffix = `_${rarity.toLowerCase()}.png`;
  return builtInImage.endsWith(suffix) && savedImage === `${builtInImage.slice(0, -suffix.length)}.png`;
}

function mergeWithBuiltIns(saved: ItemDefinition[]): ItemDefinition[] {
  const savedById = new Map(saved.map(i => [i.id, i]));
  const merged = DEFAULT_ITEMS.map(builtIn => {
    const override = savedById.get(builtIn.id);
    if (!override) return builtIn;
    savedById.delete(builtIn.id);
    // Effect wiring lives in code, so a saved copy from before effects existed
    // still resolves to the built-in's effect.
    const iconImage = isPreTintArt(builtIn.iconImage, override.iconImage, builtIn.rarity)
      ? builtIn.iconImage
      : override.iconImage ?? builtIn.iconImage;
    return { ...override, effectId: override.effectId ?? builtIn.effectId, iconImage };
  });
  return [...merged, ...Array.from(savedById.values())];
}

/**
 * Shipped items whose category changed after release. A saved copy overrides the
 * built-in wholesale, so anyone who has opened the Items editor would otherwise
 * keep the old category forever. Each id is force-reset to the built-in's
 * categories exactly once and then stamped, so a later edit in the editor sticks.
 */
const RECATEGORIZED_ITEM_IDS = ["goats_blessing"];
const RECATEGORIZED_KEY = "handz_items_recategorized";

function applyCategoryResets(items: ItemDefinition[]): ItemDefinition[] {
  if (typeof localStorage === "undefined") return items;
  let done: string[] = [];
  try {
    const parsed = JSON.parse(localStorage.getItem(RECATEGORIZED_KEY) ?? "[]");
    if (Array.isArray(parsed)) done = parsed.filter(id => typeof id === "string");
  } catch { /* a corrupt stamp just means the reset runs again, which is harmless */ }
  const pending = RECATEGORIZED_ITEM_IDS.filter(id => !done.includes(id));
  if (pending.length === 0) return items;

  const out = items.map(item => {
    if (!pending.includes(item.id)) return item;
    const builtIn = DEFAULT_ITEMS.find(d => d.id === item.id);
    if (!builtIn) return item;
    return {
      ...item,
      isBoost: builtIn.isBoost,
      isConsumable: builtIn.isConsumable,
      isKeepsake: builtIn.isKeepsake,
      obtainableLimit: builtIn.obtainableLimit,
      sellable: builtIn.sellable,
    };
  });
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(out));
    localStorage.setItem(RECATEGORIZED_KEY, JSON.stringify([...done, ...pending]));
  } catch { /* out of storage: the reset simply retries on the next load */ }
  return out;
}

export function loadItemsConfig(): ItemDefinition[] {
  try {
    const raw = typeof localStorage !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
    if (!raw) { cachedRaw = null; cachedItems = null; return DEFAULT_ITEMS; }
    if (raw === cachedRaw && cachedItems) return cachedItems;
    const parsed = JSON.parse(raw);
    const items = applyCategoryResets(mergeWithBuiltIns(Array.isArray(parsed)
      ? (parsed.map(normalizeItem).filter(Boolean) as ItemDefinition[])
      : []));
    cachedRaw = raw;
    cachedItems = items;
    return items;
  } catch {
    return DEFAULT_ITEMS;
  }
}

export function saveItemsConfig(items: ItemDefinition[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  cachedRaw = null;
  cachedItems = null;
}

export function getItemDefinition(itemId: string): ItemDefinition | undefined {
  return loadItemsConfig().find(i => i.id === itemId);
}
