#![no_main]

//! `metadata` only proves the walkers never panic; a walker that silently returns
//! `None` passes it. This builds well-formed files around a fuzzed EXIF payload,
//! padded with fuzzed neighbouring segments/chunks and JPEG fill bytes, and checks
//! the walkers hand back exactly the EXIF that went in.

use crc32fast::Hasher;
use libfuzzer_sys::fuzz_target;

#[path = "../../fcs-utils/src/output/exif.rs"]
mod exif;

/// Not a TIFF byte-order mark, so clearing Orientation leaves the payload as is.
const NOT_TIFF: &[u8] = b"XX";
/// The largest body a JPEG segment length (u16, counting itself) can describe.
const MAX_SEGMENT_BODY: usize = u16::MAX as usize - 2;

fuzz_target!(|input: (u8, Vec<(u8, Vec<u8>)>, Vec<u8>)| {
    let (fill, others, payload) = input;
    jpeg_round_trip(usize::from(fill % 8), &others, &payload);
    png_round_trip(&others, &payload);
});

fn jpeg_round_trip(fill: usize, others: &[(u8, Vec<u8>)], payload: &[u8]) {
    let mut exif_body = [b"Exif\0\0", NOT_TIFF, payload].concat();
    exif_body.truncate(MAX_SEGMENT_BODY);
    let mut expected = Vec::new();
    push_segment(&mut expected, 0, 0xE1, &exif_body);

    let mut file = vec![0xFF, 0xD8];
    for (kind, body) in others {
        // APP0, a non-Exif APP1 (XMP lives there), APP13 and COM. The leading
        // byte keeps the APP1 from ever starting with "Exif".
        let marker = [0xE0, 0xE1, 0xED, 0xFE][usize::from(*kind % 4)];
        let mut body = [b"x", body.as_slice()].concat();
        body.truncate(MAX_SEGMENT_BODY);
        push_segment(&mut file, fill, marker, &body);
    }
    push_segment(&mut file, fill, 0xE1, &exif_body);
    file.extend_from_slice(&[0xFF, 0xDA]);

    assert_eq!(exif::jpeg_exif(&file), Some(expected));
}

/// `fill` extra 0xFF bytes before the marker, which T.81 allows.
fn push_segment(file: &mut Vec<u8>, fill: usize, marker: u8, body: &[u8]) {
    file.extend(std::iter::repeat_n(0xFF, fill + 1));
    file.push(marker);
    file.extend_from_slice(&(body.len() as u16 + 2).to_be_bytes());
    file.extend_from_slice(body);
}

fn png_round_trip(others: &[(u8, Vec<u8>)], payload: &[u8]) {
    let exif = chunk(b"eXIf", &[NOT_TIFF, payload].concat());

    let mut file = b"\x89PNG\r\n\x1a\n".to_vec();
    let (before, after) = others.split_at(others.len() / 2);
    let push_others = |file: &mut Vec<u8>, chunks: &[(u8, Vec<u8>)]| {
        for (kind, data) in chunks {
            let kind = [b"IHDR", b"tEXt", b"iTXt", b"IDAT"][usize::from(*kind % 4)];
            file.extend_from_slice(&chunk(kind, data));
        }
    };
    push_others(&mut file, before);
    file.extend_from_slice(&exif);
    push_others(&mut file, after);
    file.extend_from_slice(&chunk(b"IEND", &[]));
    // Anything after IEND is not part of the image.
    file.extend_from_slice(&chunk(b"eXIf", payload));

    assert_eq!(exif::png_exif_chunks(&file), vec![exif]);
}

fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut crc = Hasher::new();
    crc.update(kind);
    crc.update(data);
    [
        &(data.len() as u32).to_be_bytes()[..],
        kind,
        data,
        &crc.finalize().to_be_bytes(),
    ]
    .concat()
}
