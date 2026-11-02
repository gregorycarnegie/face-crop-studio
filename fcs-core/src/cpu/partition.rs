//! Integer ownership calculations for parallel output rows.

/// Decode a job when groups, height and chunks are positive.
#[inline]
pub(crate) fn decode_job(
    job: usize,
    groups: usize,
    height: usize,
    chunks: usize,
) -> (usize, usize, usize, usize) {
    let row = job / chunks;
    (
        row / height / groups,
        row / height % groups,
        row % height,
        job % chunks,
    )
}

/// One member of the partition into nonempty chunks; index must be below ceil(length/chunk).
#[inline]
pub(crate) fn chunk_span(index: usize, chunk: usize, length: usize) -> (usize, usize) {
    let start = index * chunk;
    (start, start + chunk.min(length - start))
}

/// End of complete tiles in start..end, with start <= end and tile > 0.
#[inline]
pub(crate) fn tile_end(start: usize, end: usize, tile: usize) -> usize {
    start + (end - start) / tile * tile
}

/// Start of one output row in a representable NCHW allocation, for valid coordinates.
#[inline]
pub(crate) fn row_base(
    batch: usize,
    channel: usize,
    row: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> usize {
    ((batch * channels + channel) * height + row) * width
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_and_tiles_write_every_output_exactly_once() {
        for width in [1usize, 2, 3, 4, 7, 11, 49] {
            for chunk in [1usize, 2, 3, 8, 48] {
                let chunks = width.div_ceil(chunk);
                let mut visits = vec![0; 2 * 7 * 3 * width];
                for job in 0..2 * 2 * 3 * chunks {
                    let (n, group, row, part) = decode_job(job, 2, 3, chunks);
                    let (first, last) = chunk_span(group, 4, 7);
                    let (start, end) = chunk_span(part, chunk, width);
                    for channel in first..last {
                        let mut x = start;
                        let full = tile_end(start, end, 3);
                        while x < end {
                            let count = if x < full { 3 } else { 1 };
                            for p in 0..count {
                                visits[row_base(n, channel, row, 7, 3, width) + x + p] += 1;
                            }
                            x += count;
                        }
                    }
                }
                assert!(
                    visits.iter().all(|&count| count == 1),
                    "width={width}, chunk={chunk}"
                );
            }
        }
        assert_eq!(
            chunk_span(usize::MAX / 2, 2, usize::MAX),
            (usize::MAX - 1, usize::MAX)
        );
        assert_eq!(tile_end(usize::MAX - 5, usize::MAX, 3), usize::MAX - 2);
    }
}
