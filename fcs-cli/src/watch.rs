//! `--watch`: run the ordinary batch path on images as they land in a directory.
//!
//! The processing is not new -- a settled file becomes a [`ProcessingItem`] and goes
//! through the same `process_single_image` the one-shot path uses, on the same rayon
//! pool. What this module owns is deciding *when* a file is ready, which the filesystem
//! does not tell you: a create event fires as soon as the file exists, typically long
//! before whatever is writing it has finished, and decoding it then reads a truncated
//! image. So an event only puts a path on a pending list, and the path is processed once
//! its size has stopped changing for [`QUIET_PERIOD`].
//!
//! Existing files are deliberately not swept on startup. Pointing this at a folder that
//! has already been processed would otherwise rewrite every crop in the output directory
//! without being asked; `--input <dir>` is still the way to process what is already there.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{self, RecvTimeoutError},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use fcs_core::FaceDetector;
use fcs_utils::{SUPPORTED_IMAGE_EXTENSIONS, normalize_path};
use log::{info, warn};
use notify::{RecursiveMode, Watcher};
use rayon::prelude::*;

use crate::{input::ProcessingItem, workflow};

/// How long a file must stop changing before it is treated as finished.
///
/// Long enough to cover the gap between chunks of a slow copy, short enough that a
/// watched folder still feels immediate. A file that is still growing resets it.
const QUIET_PERIOD: Duration = Duration::from_millis(400);

/// How often the pending list is re-checked when no events are arriving.
const TICK: Duration = Duration::from_millis(200);

/// What a path on the pending list is waiting on: the last size seen, and when that size
/// was first seen.
type Pending = HashMap<PathBuf, (u64, Instant)>;

/// Refuse a configuration that would make the run feed on its own output.
///
/// Writing into the directory being watched is a loop: every crop lands as a new event, gets
/// detected, and produces another crop, which lands as another event. It does not converge --
/// the crop of a crop is still a face -- so the run never settles and the output directory
/// fills until someone stops it.
///
/// This is refused rather than worked around. Skipping our own writes would need the watcher to
/// know which paths came from this process, and the configuration is not one anybody wants: a
/// user asking for it has made a mistake, and a clear error beats silently ignoring half the
/// directory. `--crop` already requires `--output-dir`, so no working setup relies on this.
///
/// An output directory *inside* the watched one is fine and deliberately allowed: the watcher
/// is [`RecursiveMode::NonRecursive`], so files in a subdirectory produce no events at all.
/// Only an exact match can loop. All three paths are canonicalised by the caller, so comparing
/// them directly is sound -- `.` and a full path to the same directory compare equal.
///
/// Returns the name of the offending flag and leaves the message to the caller, which still
/// has the path as the user typed it. Canonicalisation prefixes an extended-length marker on
/// Windows, and reporting someone's output directory back to them with a `\\?\` on the front
/// helps nobody -- the same reason the watcher logs `dir` rather than `watched`.
fn self_feeding_output(
    watched: &Path,
    crop_output_dir: Option<&Path>,
    annotate_dir: Option<&Path>,
) -> Option<&'static str> {
    [
        ("--output-dir", crop_output_dir),
        ("--annotate", annotate_dir),
    ]
    .into_iter()
    .find_map(|(flag, dir)| (dir == Some(watched)).then_some(flag))
}

/// Whether a path is one this mode should pick up.
///
/// The supported list is the same one `collect_images` walks a directory with, so watch
/// mode and one-shot mode agree about what counts as an image.
fn is_watchable_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| SUPPORTED_IMAGE_EXTENSIONS.contains(&e.as_str()))
}

/// Moves every path that has stopped changing off `pending` and returns it, sorted.
///
/// A path whose size differs from the recorded one restarts its clock. A path that has
/// disappeared -- deleted, or renamed to its final name, which is how a lot of software
/// writes files -- is dropped, because the rename produces its own event for the new name.
fn take_settled(pending: &mut Pending, quiet: Duration) -> Vec<PathBuf> {
    let mut ready = Vec::new();
    pending.retain(|path, (size, since)| match std::fs::metadata(path) {
        Err(_) => false,
        Ok(meta) => {
            if meta.len() != *size {
                *size = meta.len();
                *since = Instant::now();
                return true;
            }
            if since.elapsed() >= quiet {
                ready.push(path.clone());
                return false;
            }
            true
        }
    });
    ready.sort();
    ready
}

/// Watches `dir` until the process is interrupted.
pub fn run(
    dir: &Path,
    ctx: &workflow::BatchContext<'_>,
    detector: &Arc<FaceDetector>,
    annotate_dir: &Arc<Option<PathBuf>>,
    crop_enabled: bool,
    crop_output_dir: &Arc<Option<PathBuf>>,
) -> Result<()> {
    // Canonicalised for the watcher, so event paths are absolute whatever the caller
    // passed; reported as the caller wrote it, because Windows canonicalisation prefixes
    // an extended-length marker that means nothing to the person reading the log.
    let watched = normalize_path(dir).with_context(|| format!("--watch {}", dir.display()))?;
    anyhow::ensure!(
        watched.is_dir(),
        "--watch needs a directory; {} is not one",
        dir.display()
    );
    // Before the watcher starts, not after the first crop has already triggered the next one.
    if let Some(flag) = self_feeding_output(
        &watched,
        crop_output_dir.as_deref(),
        annotate_dir.as_deref(),
    ) {
        anyhow::bail!(
            concat!(
                "{} is the directory being watched ({}), so everything written there would be ",
                "detected again and produce more files, forever. Point it outside the watched ",
                "directory, or at a subdirectory of it."
            ),
            flag,
            dir.display()
        );
    }

    let (tx, rx) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |event| {
        // The receiver lives as long as this function, so a send can only fail once
        // there is nothing left to tell.
        let _ = tx.send(event);
    })
    .context("failed to start a filesystem watcher")?;
    watcher
        .watch(&watched, RecursiveMode::NonRecursive)
        .with_context(|| format!("failed to watch {}", dir.display()))?;

    info!("Watching {} for new and changed images.", dir.display());
    info!("Files already present are not reprocessed; use --input for those. Ctrl+C to stop.");
    // Without --crop or --annotate a batch run writes nothing, and one-shot mode prints
    // its detections at the end -- which watch mode never reaches. Say so once, rather
    // than letting it look like the watcher is failing to notice files.
    if !crop_enabled && annotate_dir.is_none() {
        warn!("Neither --crop nor --annotate is set, so nothing will be written to disk.");
    }

    let mut pending = Pending::new();
    loop {
        match rx.recv_timeout(TICK) {
            Ok(Ok(event)) => {
                for path in event.paths {
                    if is_watchable_image(&path) {
                        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                        pending.entry(path).or_insert((size, Instant::now()));
                    }
                }
            }
            Ok(Err(err)) => warn!("Watcher reported an error: {err}"),
            Err(RecvTimeoutError::Timeout) => {}
            // The watcher was dropped or the backend gave up; there is nothing left to
            // wait for, so stop rather than spin.
            Err(RecvTimeoutError::Disconnected) => {
                warn!("Filesystem watcher stopped; leaving watch mode");
                break;
            }
        }

        let ready = take_settled(&mut pending, QUIET_PERIOD);
        if ready.is_empty() {
            continue;
        }
        process_batch(
            ready,
            ctx,
            detector,
            annotate_dir,
            crop_enabled,
            crop_output_dir,
        );
    }
    Ok(())
}

/// Runs one settled batch through the ordinary path and reports the running totals.
fn process_batch(
    paths: Vec<PathBuf>,
    ctx: &workflow::BatchContext<'_>,
    detector: &Arc<FaceDetector>,
    annotate_dir: &Arc<Option<PathBuf>>,
    crop_enabled: bool,
    crop_output_dir: &Arc<Option<PathBuf>>,
) {
    let items: Vec<ProcessingItem> = paths
        .into_iter()
        .map(|source| ProcessingItem {
            source,
            output_override: None,
            mapping_row: None,
        })
        .collect();
    info!("Processing {} new file(s)", items.len());

    // Same call, same pool, same counters as the one-shot path. A file that cannot be
    // decoded is logged and skipped there, which is what a watched folder wants: one bad
    // drop must not end the session.
    let handled = items
        .par_iter()
        .filter_map(|target| {
            workflow::process_single_image(
                ctx,
                target,
                detector,
                annotate_dir,
                crop_enabled,
                crop_output_dir,
            )
        })
        .count();

    let summary = ctx.counters.snapshot();
    info!(
        "Watch: {handled} of {} produced detections; totals images_processed={} faces_detected={} crops_saved={} crops_skipped_quality={}",
        items.len(),
        summary.images_processed,
        summary.faces_detected,
        summary.crops_saved,
        summary.crops_skipped_quality
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProgressCounters;
    use fcs_utils::QualityFilter;
    use std::io::Write;
    use workflow::tests::{
        batch_ctx, build_test_detector, crop_settings_app, no_gpu_runtime, parse_args,
        write_sample_png,
    };

    /// A path that is not a directory is refused before any watcher starts. `run` otherwise
    /// blocks forever, which is why nothing called it.
    #[test]
    fn watching_a_file_is_refused() {
        let Some(detector) = build_test_detector() else {
            return;
        };
        let (settings, runtime) = (crop_settings_app(), no_gpu_runtime());
        let filter = Arc::new(QualityFilter::new(None));
        let args = parse_args(&["--input", "x.jpg"]);
        let counters = ProgressCounters::default();
        let enhancement = None;
        let ctx = batch_ctx(&settings, &filter, &enhancement, &runtime, &args, &counters);

        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("not-a-dir.png");
        write_sample_png(&file);
        let err = run(
            &file,
            &ctx,
            &detector,
            &Arc::new(None),
            false,
            &Arc::new(None),
        )
        .expect_err("a file cannot be watched");
        assert!(format!("{err:#}").contains("needs a directory"), "{err:#}");
    }

    /// A settled batch goes through the ordinary per-image path and its counters.
    #[test]
    fn a_settled_batch_is_processed_like_a_one_shot_run() {
        let Some(detector) = build_test_detector() else {
            return;
        };
        let (settings, runtime) = (crop_settings_app(), no_gpu_runtime());
        let filter = Arc::new(QualityFilter::new(None));
        let args = parse_args(&["--input", "x.jpg"]);
        let counters = ProgressCounters::default();
        let enhancement = None;
        let ctx = batch_ctx(&settings, &filter, &enhancement, &runtime, &args, &counters);

        let dir = tempfile::tempdir().expect("tempdir");
        let image = dir.path().join("dropped.png");
        write_sample_png(&image);
        process_batch(
            vec![image],
            &ctx,
            &detector,
            &Arc::new(None),
            false,
            &Arc::new(None),
        );
        assert_eq!(counters.snapshot().images_processed, 1);
    }

    #[test]
    fn only_supported_image_extensions_are_picked_up() {
        assert!(is_watchable_image(Path::new("a/b/photo.JPG")));
        assert!(is_watchable_image(Path::new("photo.png")));
        assert!(!is_watchable_image(Path::new("photo.jpg.part")));
        assert!(!is_watchable_image(Path::new("notes.txt")));
        assert!(!is_watchable_image(Path::new("no_extension")));
    }

    /// The point of the pending list: a file that is still being written must not be
    /// handed to the decoder, and one that has stopped must not be held forever.
    #[test]
    fn a_growing_file_waits_and_a_settled_one_is_released() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("incoming.jpg");
        let mut file = std::fs::File::create(&path).expect("create");
        file.write_all(b"first chunk").expect("write");
        file.flush().expect("flush");

        let mut pending = Pending::new();
        let size = std::fs::metadata(&path).expect("stat").len();
        pending.insert(path.clone(), (size, Instant::now()));

        // Still growing: the size changed, so the clock restarts and nothing is released
        // even with a quiet period of zero.
        file.write_all(b" second chunk").expect("write");
        file.flush().expect("flush");
        assert!(
            take_settled(&mut pending, Duration::ZERO).is_empty(),
            "a file that just changed size must not be released"
        );
        assert_eq!(pending.len(), 1, "and it must stay on the list");

        // Unchanged since the previous check, and the quiet period has passed.
        assert_eq!(take_settled(&mut pending, Duration::ZERO), vec![path]);
        assert!(pending.is_empty(), "a released file is taken off the list");
    }

    #[test]
    fn a_file_that_disappears_is_dropped_rather_than_retried() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("gone.jpg");
        let mut pending = Pending::new();
        pending.insert(path, (0, Instant::now()));

        assert!(take_settled(&mut pending, Duration::ZERO).is_empty());
        assert!(
            pending.is_empty(),
            "nothing is left waiting on a missing file"
        );
    }

    /// A file still inside its quiet period is neither released nor forgotten.
    #[test]
    fn a_settled_file_is_held_until_the_quiet_period_elapses() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("held.jpg");
        std::fs::write(&path, b"complete").expect("write");

        let mut pending = Pending::new();
        let size = std::fs::metadata(&path).expect("stat").len();
        pending.insert(path.clone(), (size, Instant::now()));

        assert!(take_settled(&mut pending, Duration::from_secs(30)).is_empty());
        assert_eq!(pending.len(), 1);
        assert_eq!(take_settled(&mut pending, Duration::ZERO), vec![path]);
    }

    /// The loop this guards against: crops landing in the watched directory are detected again.
    #[test]
    fn an_output_dir_equal_to_the_watched_dir_is_refused() {
        let watched = Path::new("/data/incoming");

        assert_eq!(
            self_feeding_output(watched, Some(watched), None),
            Some("--output-dir"),
            "--output-dir into the watched directory must be refused"
        );
        assert_eq!(
            self_feeding_output(watched, None, Some(watched)),
            Some("--annotate"),
            "--annotate into the watched directory must be refused"
        );
    }

    /// A subdirectory cannot loop, because the watcher is non-recursive, so it must be allowed
    /// -- refusing it would reject the obvious `--watch in --output-dir in/crops` layout.
    #[test]
    fn an_output_dir_below_or_outside_the_watched_dir_is_allowed() {
        let watched = Path::new("/data/incoming");
        for out in [
            Path::new("/data/incoming/crops"),
            Path::new("/data/outgoing"),
            Path::new("/elsewhere"),
        ] {
            assert_eq!(
                self_feeding_output(watched, Some(out), None),
                None,
                "{} should be allowed",
                out.display()
            );
        }
        // And with neither flag set there is nothing to check.
        assert_eq!(self_feeding_output(watched, None, None), None);
    }
}
