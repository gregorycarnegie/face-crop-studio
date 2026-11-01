#[path = "../../fcs-core/src/cropper/region.rs"]
pub mod region;

#[cfg(test)]
mod tests {
    use super::region::CropRegion;

    #[test]
    fn intersection_matches_geometry_at_integer_boundaries() {
        let coordinates = [i32::MIN, -100, -1, 0, 1, 100, i32::MAX];
        let sizes = [0, 1, 2, 100, i32::MAX as u32, i32::MAX as u32 + 1, u32::MAX];
        for x in coordinates {
            for y in coordinates {
                for width in sizes {
                    for height in sizes {
                        for img_w in sizes {
                            for img_h in sizes {
                                let crop = CropRegion::new(x, y, width, height, img_w, img_h);
                                let left = i64::from(x).max(0);
                                let top = i64::from(y).max(0);
                                let right = (i64::from(x) + i64::from(width)).min(i64::from(img_w));
                                let bottom =
                                    (i64::from(y) + i64::from(height)).min(i64::from(img_h));
                                let expected = if right <= left || bottom <= top {
                                    None
                                } else {
                                    Some((
                                        left as u32,
                                        top as u32,
                                        (right - left) as u32,
                                        (bottom - top) as u32,
                                    ))
                                };
                                assert_eq!(
                                    crop.in_bounds_rect(img_w, img_h),
                                    expected,
                                    "{crop:?}, image={img_w}x{img_h}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
