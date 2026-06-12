//! Helpers for exporting cropped images with flexible encoding and metadata handling.
//!
//! This module centralizes output-format selection, compression tuning, and metadata
//! preservation so that both the CLI and GUI can share a single implementation.

mod encoders;
mod metadata;
#[cfg(test)]
mod tests;
mod types;
mod writer;

pub use types::{ImageFormatHint, MetadataContext, OutputOptions, PngCompression};
pub use writer::{append_suffix_to_filename, save_dynamic_image};

#[cfg(test)]
use {
    crate::{
        config::{MetadataMode, MetadataSettings},
        quality::Quality,
    },
    crc32fast::Hasher as Crc32,
    encoders::{encode_jpeg, encode_png},
    metadata::{
        build_custom_metadata_payload, inject_jpeg_metadata, inject_png_metadata, load_jpeg_exif,
        load_png_exif_chunks,
    },
    std::{fs, path::Path},
    writer::determine_format,
};
