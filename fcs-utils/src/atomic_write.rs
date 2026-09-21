//! Replacing a file without destroying it on the way.
//!
//! Every write this app does is a *replacement*: a crop over a previous export, the settings
//! file over the last one, a detection JSON over yesterday's. The obvious spelling of that,
//! `File::create` or `fs::write`, truncates the destination before the new bytes are written,
//! so a write that fails part-way has already destroyed what was there. On a full disk the
//! result is an empty file where a good export used to be.
//!
//! Two of these were live and reported success while doing it; see [`write_atomically`].

use anyhow::{Context, Result};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// A temporary file that deletes itself unless [`Self::keep`] is called.
///
/// Without it, a failed write or a panic mid-write would leave `fcs-tmp1234-0` sitting in the
/// user's output directory. The destination is untouched either way, which is the point.
struct TempFile {
    path: PathBuf,
    keep: bool,
}

impl TempFile {
    /// A unique name in `dir`. The counter covers the rayon workers writing one batch, and the
    /// pid covers two runs sharing an output directory.
    fn new(dir: &Path) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            // Deliberately short, and not derived from the destination: output names are
            // already long, and prefixing one risks passing the 255-byte filename limit.
            path: dir.join(format!("fcs-tmp{}-{n}", std::process::id())),
            keep: false,
        }
    }

    /// The file has been renamed away, so there is nothing left to clean up.
    fn keep(mut self) {
        self.keep = true;
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Write `bytes` to `path`, leaving an existing file intact unless the write fully succeeds.
///
/// Two failures this exists to avoid, both of which were live in shipped builds:
///
/// * `BufWriter::write_all` only fills a buffer, so an out-of-space or IO error surfaces at
///   `flush` -- whose result the crop writer discarded with `.ok()`. A failed export was
///   reported as a successful one, by the CLI summary and the GUI alike.
/// * `File::create` and `fs::write` truncate immediately, so a write that then failed left a
///   truncated or empty file where good data had been. **The previous file was destroyed by the
///   attempt to replace it** -- including, for the settings writer, the user's whole config.
///
/// So the bytes go to a temporary file beside the destination -- the same directory, so the
/// rename cannot cross a filesystem -- and replace it only once every byte is written.
/// `fs::rename` over an existing file replaces it in one step on NTFS and POSIX alike.
///
/// This makes the *replacement* atomic, not durable: there is no `fsync`, so a power loss can
/// still lose a just-written file. Adding one would cost a flush-to-disk per crop across a whole
/// batch, and the failure this fixes is an error being reported as success.
///
/// Parent directories are not created here; the caller decides whether a missing directory is
/// an error or something to create.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    // A destination with no parent is a bare filename, which lands in the working directory.
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let temp = TempFile::new(parent);

    {
        let file = File::create(&temp.path).with_context(|| {
            format!("failed to create a temporary file in {}", parent.display())
        })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(bytes)
            .with_context(|| format!("failed to write {}", path.display()))?;
        // The error `.ok()` used to swallow: for anything larger than the buffer, this is where
        // a full disk or a failing drive actually reports itself.
        writer
            .flush()
            .with_context(|| format!("failed to flush {}", path.display()))?;
    }

    fs::rename(&temp.path, path).with_context(|| {
        format!(
            "failed to move the finished file into place at {}",
            path.display()
        )
    })?;
    temp.keep();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn writes_a_new_file() {
        let dir = tempdir().expect("tempdir");
        let dest = dir.path().join("out.bin");
        write_atomically(&dest, b"hello").expect("write");
        assert_eq!(fs::read(&dest).expect("read"), b"hello");
    }

    #[test]
    fn replaces_an_existing_file() {
        let dir = tempdir().expect("tempdir");
        let dest = dir.path().join("out.bin");
        fs::write(&dest, b"old contents, longer than the new").expect("seed");
        write_atomically(&dest, b"new").expect("write");
        assert_eq!(fs::read(&dest).expect("read"), b"new");
    }

    /// Nothing may be left behind in the output directory, or a batch would litter one temp
    /// file per crop next to the crops.
    #[test]
    fn leaves_no_temporary_file_behind_on_success() {
        let dir = tempdir().expect("tempdir");
        let dest = dir.path().join("out.bin");
        write_atomically(&dest, b"hello").expect("write");
        let names: Vec<_> = fs::read_dir(dir.path())
            .expect("read_dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["out.bin".to_string()]);
    }

    /// The point of the whole module: a write that fails must not take the old file with it.
    ///
    /// Forcing a *mid-write* failure portably is not possible -- there is no way to fill the
    /// disk from a test -- so this forces the failure at the final rename instead, by pointing
    /// the destination at a directory. `fs::rename` refuses to replace a directory with a file
    /// on both Windows and POSIX, which exercises the same error path a full disk would reach,
    /// and the file inside it stands in for the data that must survive.
    #[test]
    fn a_failed_write_preserves_what_was_there() {
        let dir = tempdir().expect("tempdir");
        let occupied = dir.path().join("occupied");
        fs::create_dir(&occupied).expect("mkdir");
        fs::write(occupied.join("precious.txt"), b"must survive").expect("seed");

        let err = write_atomically(&occupied, b"new bytes").expect_err("must refuse");
        assert!(
            format!("{err:#}").contains("occupied"),
            "the error should name the destination: {err:#}"
        );

        assert_eq!(
            fs::read(occupied.join("precious.txt")).expect("read"),
            b"must survive",
            "the existing data was destroyed by a failed write"
        );
        // And the temp file is cleaned up even on the failure path.
        let strays: Vec<_> = fs::read_dir(dir.path())
            .expect("read_dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .filter(|name| name != "occupied")
            .collect();
        assert!(strays.is_empty(), "left behind: {strays:?}");
    }

    /// A bare filename has no parent component, which must mean the working directory rather
    /// than an empty path that no filesystem accepts.
    #[test]
    fn a_bare_filename_resolves_to_the_working_directory() {
        let dir = tempdir().expect("tempdir");
        let restore = std::env::current_dir().expect("cwd");
        // `set_current_dir` is process-wide, so this test must not run beside another that
        // depends on the working directory. Nothing else here does.
        std::env::set_current_dir(dir.path()).expect("chdir");
        let result = write_atomically(Path::new("bare.bin"), b"x");
        let written = fs::read(dir.path().join("bare.bin"));
        std::env::set_current_dir(restore).expect("restore cwd");

        result.expect("write");
        assert_eq!(written.expect("read"), b"x");
    }
}
