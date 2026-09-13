//! Its own test binary: `init_logging` installs the process-wide logger, and the unit tests in
//! `telemetry.rs` rely on no logger being installed in theirs.

use log::LevelFilter;

#[test]
fn init_logging_installs_a_logger_with_telemetry_at_trace() {
    fcs_utils::init_logging(LevelFilter::Warn).unwrap();
    // The fcs::telemetry module filter raises the global ceiling to Trace; with no logger
    // installed it stays Off.
    assert_eq!(log::max_level(), LevelFilter::Trace);
}
