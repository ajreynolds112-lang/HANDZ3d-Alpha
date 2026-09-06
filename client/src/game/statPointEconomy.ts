/**
 * The stat-point economy: what a bought stat point costs, and what an earned
 * one is worth once the cap leaves no room for it.
 *
 * Both halves are driven by counters that live in the career save, so they
 * travel with an exported file and start clean in a brand-new career.
 */

/** Force price of the very first stat point a career buys. */
export const STAT_POINT_BASE_COST = 10_000;
/** Every purchase makes the next one 5% dearer. */
export const STAT_POINT_COST_MULT = 1.05;

/**
 * Force needed for the next bought stat point, given how many this career has
 * already bought. The price is driven by purchases alone — points earned in
 * the ring or the gym never make the next one dearer.
 */
export function statPointForceCost(pointsBought: number | null | undefined): number {
  // Saves written before paid stat points existed carry no counter — they have
  // bought nothing, so they start at the base price rather than NaN.
  const bought = Number.isFinite(pointsBought) ? Math.max(0, pointsBought as number) : 0;
  const raw = STAT_POINT_BASE_COST * Math.pow(STAT_POINT_COST_MULT, bought);
  // Thousands of purchases in, the curve overflows a double — clamp it.
  if (!Number.isFinite(raw) || raw > Number.MAX_SAFE_INTEGER) return Number.MAX_SAFE_INTEGER;
  return Math.ceil(raw);
}

/** Force paid per stat point an official bout earned past the cap. */
export const FORCE_PER_CAPPED_SP_BOUT = 100_000;
/** Force paid per stat point a training session earned past the cap. */
export const FORCE_PER_CAPPED_SP_TRAINING = 10_000;

/**
 * What the points that could not fit under the stat cap are worth instead.
 * Every reward path clamps its earned points to the remaining headroom; the
 * difference is handed back as Force rather than being lost.
 */
export function cappedStatPointForce(overflowPoints: number, ratePerPoint: number): number {
  const pts = Number.isFinite(overflowPoints) ? Math.max(0, Math.floor(overflowPoints)) : 0;
  return pts * ratePerPoint;
}
