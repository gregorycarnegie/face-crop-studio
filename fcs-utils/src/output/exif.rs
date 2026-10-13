//! EXIF byte walkers shared by file loading and the fuzz targets.

use crc32fast::Hasher as Crc32;
use image::metadata::Orientation;

pub(super) fn png_exif_chunks(bytes: &[u8]) -> Vec<Vec<u8>> {
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Vec::new();
    }

    let mut cursor = 8usize;
    let mut chunks = Vec::new();
    while cursor + 8 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        let chunk_type = &bytes[cursor + 4..cursor + 8];
        let data_start = cursor + 8;
        let data_end = data_start.saturating_add(length);
        // `data_end` saturates, so `data_end + 4` could overflow on 32-bit targets.
        if data_end > bytes.len().saturating_sub(4) {
            break;
        }

        if chunk_type == b"eXIf" {
            chunks.push(exif_chunk_without_orientation(&bytes[data_start..data_end]));
        }

        cursor = data_end + 4;
        if chunk_type == b"IEND" {
            break;
        }
    }
    chunks
}

pub(super) fn jpeg_exif(bytes: &[u8]) -> Option<Vec<u8>> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }

    let mut index = 2usize;
    while index + 4 < bytes.len() {
        if bytes[index] != 0xFF {
            break;
        }
        // Any number of 0xFF fill bytes may precede a marker (T.81 B.1.1.2).
        while bytes[index + 1] == 0xFF {
            index += 1;
            if index + 4 >= bytes.len() {
                return None;
            }
        }
        let marker = bytes[index + 1];
        index += 2;

        if marker == 0xDA || marker == 0xD9 {
            break;
        }

        // `index + 4 < bytes.len()` held before the marker consumed two bytes
        // (the loop condition, re-checked after each fill byte), so the two
        // length bytes below are always in range.
        let length = u16::from_be_bytes([bytes[index], bytes[index + 1]]) as usize;
        // A JPEG segment length counts its own two length bytes, so anything
        // below 2 is malformed. Subtracting first underflowed: a crafted file
        // with a zero length panicked in debug builds and wrapped to a huge
        // offset in release ones. Bail instead of trusting the value.
        if length < 2 {
            break;
        }
        let data_start = index + 2;
        let data_end = data_start.saturating_add(length - 2);
        if data_end > bytes.len() {
            break;
        }

        if marker == 0xE1 && length >= 8 && &bytes[data_start..data_start + 4] == b"Exif" {
            let mut segment = Vec::with_capacity(length + 2);
            segment.extend_from_slice(&[0xFF, 0xE1]);
            segment.extend_from_slice(&bytes[index..index + length]);
            clear_jpeg_exif_orientation(&mut segment);
            return Some(segment);
        }

        index = data_end;
    }
    None
}

/// A whole `eXIf` chunk for `data` with its Orientation tag cleared, and the CRC recomputed.
///
/// `load_image` has already rotated the pixels upright, so copying the original instruction
/// made viewers rotate the export a second time. The JPEG path has always cleared it.
fn exif_chunk_without_orientation(data: &[u8]) -> Vec<u8> {
    let mut data = data.to_vec();
    let _ = Orientation::remove_from_exif_chunk(&mut data);
    let mut hasher = Crc32::new();
    hasher.update(b"eXIf");
    hasher.update(&data);
    let mut chunk = Vec::with_capacity(12 + data.len());
    chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
    chunk.extend_from_slice(b"eXIf");
    chunk.extend_from_slice(&data);
    chunk.extend_from_slice(&hasher.finalize().to_be_bytes());
    chunk
}

/// `segment` is a whole APP1 segment: marker (2), length (2), then `Exif\0\0` and the TIFF body.
/// Its only caller builds exactly that, so only the Exif header needs checking.
fn clear_jpeg_exif_orientation(segment: &mut [u8]) {
    if segment.get(4..10) == Some(b"Exif\0\0".as_slice()) {
        let _ = Orientation::remove_from_exif_chunk(&mut segment[10..]);
    }
}
