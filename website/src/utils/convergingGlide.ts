import { CONVERGE_MAX_MS } from './searchScroll'

/**
 * A programmatic scroll that is SMOOTH and lands EXACTLY, on a scroller whose
 * geometry is still moving while it scrolls — the pinned-prompt jump.
 *
 * Two facts make a native `scrollTo({ behavior: 'smooth' })` the wrong tool:
 *
 * - Any other `scrollTop` write cancels a native animation where it stands, and
 *   writes DO land mid-flight: the virtualizer's anchor compensation as rows
 *   above the viewport measure in, its height-sync compensation, a re-measuring
 *   row. Each strands the scroll part-way — the "moved a few hundred pixels and
 *   stopped" report.
 * - The destination itself moves. Rows mount and measure as the viewport passes
 *   them, images load, and the banner the landing must clear swaps the moment
 *   the target un-pins — a swap CAUSED by the landing write, so it is only
 *   visible on the frame after. A single computed destination is therefore
 *   stale by the time it is reached.
 *
 * So the glide owns every frame's write (uncancellable by other writers) and
 * re-derives its destination from live geometry every frame. It runs in two
 * phases:
 *
 * 1. TRAVEL — an eased interpolation from the starting position to the LIVE
 *    goal, over `durationMs`. Because the goal is re-read each frame, drift
 *    during travel is absorbed into the remaining motion rather than left over.
 * 2. CONVERGE — after travel, keep writing the live goal until it has held
 *    still for `quietMs` (and at least two frames), or the backstop expires.
 *    This is what catches the post-landing shifts: the banner swap, a late
 *    image, a row that measured after the last travel frame. Travel alone
 *    landed on the goal as it was on its final frame, which is exactly the
 *    reading that the landing then invalidates.
 *
 * Reduced motion skips travel, not convergence: the reader asked for no eased
 * motion, not for an inexact landing.
 *
 * Pure: clock, frame scheduler and every DOM read/write are injected, so the
 * two-phase contract is unit-testable without a live scroller.
 */

/** Shortest travel, for a jump within a couple of viewports. */
export const GLIDE_MIN_MS = 450
/** Longest travel: a jump across tens of thousands of pixels still reads as a
 *  scroll rather than a blur, without making the reader wait on it. */
export const GLIDE_MAX_MS = 900
/** Travel speed that scales the duration between the two bounds. */
export const GLIDE_PX_PER_MS = 24

/**
 * Time a converging glide holds still before it is settled. Long enough to
 * outlast the banner swap and a row measuring on the frame after landing (both
 * one or two frames), short enough that the reader never notices the hold.
 * Deliberately shorter than `MIN_QUIET_MS`: that window waits out a widget
 * iframe build for a jump INTO a widget, and a pinned prompt is never one.
 */
export const GLIDE_QUIET_MS = 250

/** Travel duration for a jump of `distancePx`, clamped to the bounds above. */
export function glideDurationMs(distancePx: number): number {
  const byDistance = Math.abs(distancePx) / GLIDE_PX_PER_MS
  return Math.min(GLIDE_MAX_MS, Math.max(GLIDE_MIN_MS, Math.round(byDistance)))
}

export type GlideEnd = 'settled' | 'timeout' | 'cancelled' | 'lost'

export interface ConvergingGlideDeps {
  /**
   * Live destination in scroller `scrollTop` pixels, or `null` when it cannot be
   * derived this frame (the target row is gone and no estimate exists). A null
   * goal ends the glide with reason `lost` — there is nothing to converge on.
   */
  goal: () => number | null
  /** Current `scrollTop`. */
  read: () => number
  /** Write `scrollTop`. Called at most once per frame. */
  write: (top: number) => void
  /** Travel duration; see `glideDurationMs`. Ignored when `reduced`. */
  durationMs: number
  /** Skip the eased travel (prefers-reduced-motion). Convergence still runs. */
  reduced?: boolean
  /** Quiet window before the goal counts as settled. Default `GLIDE_QUIET_MS`. */
  quietMs?: number
  /** Wall-clock backstop for the CONVERGE phase, timed from its start. */
  convergeMaxMs?: number
  now?: () => number
  raf?: (cb: () => void) => number
  cancelRaf?: (id: number) => void
  onEnd?: (reason: GlideEnd) => void
}

const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3)

/**
 * Start a converging glide. Returns a cancel function — idempotent, and a no-op
 * after the glide has ended on its own.
 */
export function runConvergingGlide(deps: ConvergingGlideDeps): () => void {
  const {
    goal,
    read,
    write,
    durationMs,
    reduced = false,
    quietMs = GLIDE_QUIET_MS,
    convergeMaxMs = CONVERGE_MAX_MS,
    now = () => performance.now(),
    raf = (cb) => requestAnimationFrame(cb),
    cancelRaf = (id) => cancelAnimationFrame(id),
  } = deps
  const t0 = now()
  const from = read()
  let done = false
  let frameId = 0
  // CONVERGE bookkeeping. `convergeStart` is set on the first converge frame so
  // the backstop is timed from the end of travel, whatever travel took.
  let convergeStart: number | null = null
  let lastGoal: number | null = null
  let quietFrames = 0
  let quietSince = 0
  const finish = (reason: GlideEnd) => {
    if (done) return
    done = true
    cancelRaf(frameId)
    deps.onEnd?.(reason)
  }
  const frame = () => {
    if (done) return
    const g = goal()
    if (g == null) return finish('lost')
    const t = reduced ? 1 : Math.min(1, (now() - t0) / durationMs)
    if (t < 1) {
      // TRAVEL. The goal is live, so the interpolation's endpoint moves with it
      // and the residual is folded into the remaining motion.
      write(from + (g - from) * easeOutCubic(t))
      frameId = raf(frame)
      return
    }
    // CONVERGE. Write the live goal every frame until it stops moving.
    write(g)
    const at = now()
    if (convergeStart == null) convergeStart = at
    if (lastGoal != null && Math.abs(g - lastGoal) < 1) {
      quietFrames += 1
      if (quietFrames >= 2 && at - quietSince >= quietMs) return finish('settled')
    } else {
      quietFrames = 0
      quietSince = at
    }
    lastGoal = g
    if (at - convergeStart >= convergeMaxMs) return finish('timeout')
    frameId = raf(frame)
  }
  frameId = raf(frame)
  return () => finish('cancelled')
}
