//! Lightweight timing utilities for optional performance tracing.
//!
//! The helpers in this module provide a simple RAII guard that records the
//! elapsed duration of a scoped operation and logs it when the guard is dropped.
//! Logging only occurs when both the requested log level is enabled and the
//! caller explicitly opts in (via [`timing_guard_if`]). This keeps the overhead
//! negligible when tracing is disabled.

use log::{Level, LevelFilter, log, log_enabled};
use std::{
    borrow::Cow,
    sync::atomic::{AtomicBool, AtomicU8, Ordering},
    time::{Duration, Instant},
};

static TELEMETRY_ENABLED: AtomicBool = AtomicBool::new(false);
static TELEMETRY_LEVEL: AtomicU8 = AtomicU8::new(LevelFilter::Off as u8);

/// RAII helper that logs how long an operation took when dropped.
///
/// Guards are usually created via [`timing_guard`] or [`timing_guard_if`] so
/// most callers do not need to interact with this type directly.
pub struct TimingGuard {
    label: Cow<'static, str>,
    level: Level,
    start: Instant,
    active: bool,
}

impl TimingGuard {
    /// Create a guard with an explicit activation flag.
    fn new(label: Cow<'static, str>, level: Level, active: bool) -> Self {
        Self {
            label,
            level,
            start: Instant::now(),
            active,
        }
    }

    /// Returns `true` when the guard will emit a log entry on drop.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Returns the elapsed duration since the guard was created.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Consume the guard and return the elapsed duration without logging.
    pub fn finish(mut self) -> Duration {
        let duration = self.start.elapsed();
        self.active = false;
        duration
    }
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        if self.active {
            let duration = self.start.elapsed();
            log!(
                target: "fcs::telemetry",
                self.level,
                "{} completed in {:.2?}",
                self.label,
                duration
            );
        }
    }
}

/// Create a timing guard that logs at the provided level when that level is enabled.
///
/// Logging only occurs when the global logger allows the provided level (e.g. via
/// `RUST_LOG=fcs=debug`). This is the preferred helper when the guard should
/// activate automatically based on the current log filter.
pub fn timing_guard(label: impl Into<Cow<'static, str>>, level: Level) -> TimingGuard {
    timing_guard_if(label, level, true)
}

/// Create a timing guard that also respects an explicit boolean flag.
///
/// This variant gives callers the ability to toggle telemetry at runtime (e.g.
/// via configuration) in addition to the global log filter.
pub fn timing_guard_if(
    label: impl Into<Cow<'static, str>>,
    level: Level,
    enabled: bool,
) -> TimingGuard {
    let label = label.into();
    let active =
        enabled && telemetry_allows(level) && log_enabled!(target: "fcs::telemetry", level);
    TimingGuard::new(label, level, active)
}

/// Configure the global telemetry state.
///
/// Callers should invoke this whenever user preferences change so guards can
/// pick up the new settings.
pub fn configure(enabled: bool, level: LevelFilter) {
    TELEMETRY_ENABLED.store(enabled, Ordering::Relaxed);
    TELEMETRY_LEVEL.store(filter_index(level), Ordering::Relaxed);
}

/// Returns whether telemetry logging is currently enabled.
pub fn telemetry_enabled() -> bool {
    TELEMETRY_ENABLED.load(Ordering::Relaxed)
}

/// Returns the maximum telemetry logging level.
pub fn telemetry_level() -> LevelFilter {
    filter_from_index(TELEMETRY_LEVEL.load(Ordering::Relaxed))
}

/// Returns `true` when telemetry is enabled and the provided level is within
/// the configured threshold.
pub fn telemetry_allows(level: Level) -> bool {
    if !telemetry_enabled() {
        return false;
    }
    let threshold = TELEMETRY_LEVEL.load(Ordering::Relaxed);
    level_index(level) <= threshold
}

fn level_index(level: Level) -> u8 {
    match level {
        Level::Error => 1,
        Level::Warn => 2,
        Level::Info => 3,
        Level::Debug => 4,
        Level::Trace => 5,
    }
}

fn filter_index(filter: LevelFilter) -> u8 {
    match filter {
        LevelFilter::Off => 0,
        LevelFilter::Error => 1,
        LevelFilter::Warn => 2,
        LevelFilter::Info => 3,
        LevelFilter::Debug => 4,
        LevelFilter::Trace => 5,
    }
}

fn filter_from_index(value: u8) -> LevelFilter {
    match value {
        1 => LevelFilter::Error,
        2 => LevelFilter::Warn,
        3 => LevelFilter::Info,
        4 => LevelFilter::Debug,
        5 => LevelFilter::Trace,
        _ => LevelFilter::Off,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reset global telemetry state between tests so they don't interfere.
    fn reset() {
        configure(false, LevelFilter::Off);
    }

    #[test]
    fn configure_roundtrips_enabled_flag_and_level() {
        configure(true, LevelFilter::Debug);
        assert!(telemetry_enabled());
        assert_eq!(telemetry_level(), LevelFilter::Debug);

        configure(false, LevelFilter::Off);
        assert!(!telemetry_enabled());
        assert_eq!(telemetry_level(), LevelFilter::Off);
    }

    #[test]
    fn telemetry_allows_respects_enabled_flag() {
        reset();
        // Disabled → never allowed regardless of level.
        assert!(!telemetry_allows(Level::Error));

        configure(true, LevelFilter::Info);
        assert!(telemetry_allows(Level::Error));
        assert!(telemetry_allows(Level::Warn));
        assert!(telemetry_allows(Level::Info));
        assert!(!telemetry_allows(Level::Debug));
        assert!(!telemetry_allows(Level::Trace));

        reset();
    }

    #[test]
    fn telemetry_allows_all_levels_at_trace() {
        configure(true, LevelFilter::Trace);
        for level in [
            Level::Error,
            Level::Warn,
            Level::Info,
            Level::Debug,
            Level::Trace,
        ] {
            assert!(telemetry_allows(level), "expected {level:?} to be allowed");
        }
        reset();
    }

    #[test]
    fn timing_guard_finish_suppresses_log_and_returns_duration() {
        // finish() should mark the guard inactive so Drop does not log.
        let guard = timing_guard_if("test_op", Level::Debug, false);
        assert!(!guard.is_active());
        let elapsed = guard.finish();
        // We can't assert an exact value, but it should be a valid Duration.
        assert!(elapsed.as_nanos() < 1_000_000_000); // < 1 second
    }

    #[test]
    fn timing_guard_elapsed_increases_over_time() {
        let guard = timing_guard_if("op", Level::Debug, false);
        let d1 = guard.elapsed();
        // Spin briefly to ensure time advances.
        let start = std::time::Instant::now();
        while start.elapsed().as_nanos() < 1_000 {}
        let d2 = guard.elapsed();
        assert!(d2 >= d1);
        drop(guard);
    }

    #[test]
    fn timing_guard_if_inactive_when_disabled() {
        let guard = timing_guard_if("op", Level::Debug, false);
        assert!(!guard.is_active());
    }

    // ------------------------------------------------------------------
    // The guard's own accessors, and the numeric level mapping, had no
    // coverage: `elapsed`, `finish` and `Drop` could all be replaced with
    // no-ops or defaults without any test noticing.
    // ------------------------------------------------------------------

    #[test]
    fn elapsed_advances_and_finish_returns_a_real_duration() {
        reset();
        let guard = timing_guard_if("op", Level::Error, false);

        // Busy-wait rather than sleep: this only needs the clock to move, and a
        // sleep would make the test slow for no extra signal.
        let spin_until_measurable = || {
            let t = std::time::Instant::now();
            while t.elapsed() == Duration::ZERO {
                std::hint::spin_loop();
            }
        };
        spin_until_measurable();

        let first = guard.elapsed();
        assert!(
            first > Duration::ZERO,
            "elapsed must reflect real time, got {first:?}"
        );

        spin_until_measurable();
        assert!(
            guard.elapsed() >= first,
            "elapsed must be monotonic non-decreasing"
        );

        let finished = guard.finish();
        assert!(
            finished >= first,
            "finish must report at least what elapsed already reported: {finished:?} vs {first:?}"
        );
    }

    /// `timing_guard_if` can never yield an active guard under `cargo test`: its
    /// third condition is `log_enabled!`, and no logger is installed, so that is
    /// always false. This is why every other test here only checks inactive
    /// cases. To reach the active path at all, build the guard with the private
    /// constructor that `timing_guard_if` itself calls.
    #[test]
    fn finish_reports_a_duration_and_the_active_flag_round_trips() {
        reset();

        let guard = TimingGuard::new("op".into(), Level::Error, true);
        assert!(
            guard.is_active(),
            "a guard constructed active must report itself active"
        );
        let elapsed = guard.finish();
        assert!(
            elapsed < Duration::from_secs(1),
            "finish should report this scope's own duration, got {elapsed:?}"
        );

        let inactive = TimingGuard::new("quiet".into(), Level::Error, false);
        assert!(!inactive.is_active());

        // Dropping an active guard is the logging path. With no logger installed
        // it has no observable effect, so this only asserts it does not panic.
        drop(TimingGuard::new("dropped".into(), Level::Error, true));

        reset();
    }

    #[test]
    fn timing_guard_if_requires_every_condition_to_hold() {
        // `active` is `enabled && telemetry_allows(level) && log_enabled!(..)`.
        // An `||` in place of either `&&` would activate the guard when only one
        // condition held, so check the two failing combinations independently.
        reset();

        // Telemetry configured off: even an explicitly enabled guard is inert.
        configure(false, LevelFilter::Off);
        assert!(
            !timing_guard_if("op", Level::Error, true).is_active(),
            "telemetry disabled globally must win over enabled=true"
        );

        // Telemetry on, but the caller passed enabled=false.
        configure(true, LevelFilter::Trace);
        assert!(
            !timing_guard_if("op", Level::Error, false).is_active(),
            "enabled=false must win over telemetry being on"
        );

        // Telemetry on but the level is above the configured filter.
        configure(true, LevelFilter::Error);
        assert!(
            !timing_guard_if("op", Level::Trace, true).is_active(),
            "a level the filter excludes must not activate"
        );

        // Every telemetry-side condition satisfied — enabled, and the level well
        // inside the filter — yet still inactive, because no logger is installed
        // so `log_enabled!` is false. This is the case that separates the second
        // `&&` from an `||`: with `||` the guard would read
        // `enabled && (telemetry_allows || log_enabled)` and activate here.
        configure(true, LevelFilter::Trace);
        assert!(
            !timing_guard_if("op", Level::Error, true).is_active(),
            "log_enabled! is false without a logger, so the guard must stay \
             inactive even with telemetry fully enabled"
        );

        reset();
    }

    #[test]
    fn filter_and_index_round_trip_for_every_level() {
        // Each arm is deleted individually by mutation, so assert each mapping
        // rather than only the endpoints.
        let pairs = [
            (1u8, LevelFilter::Error),
            (2, LevelFilter::Warn),
            (3, LevelFilter::Info),
            (4, LevelFilter::Debug),
            (5, LevelFilter::Trace),
        ];
        for (index, filter) in pairs {
            assert_eq!(filter_from_index(index), filter, "index {index}");
            assert_eq!(filter_index(filter), index, "filter {filter:?}");
        }

        // Everything outside 1..=5 is Off, including 0 and the far end of u8.
        for index in [0u8, 6, 7, 100, u8::MAX] {
            assert_eq!(
                filter_from_index(index),
                LevelFilter::Off,
                "index {index} should be Off"
            );
        }
        assert_eq!(filter_index(LevelFilter::Off), 0);
    }
}
