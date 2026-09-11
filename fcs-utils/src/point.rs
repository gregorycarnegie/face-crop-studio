use std::ops::{Add, Div, Mul, Neg, Sub};

/// Single 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Horizontal coordinate in the caller's coordinate system.
    pub x: f32,
    /// Vertical coordinate in the caller's coordinate system.
    pub y: f32,
}

impl Point {
    /// Create a new point with the given coordinates.
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Compute `self * a + b` with a fused multiply-add for each coordinate.
    pub fn mul_add(self, a: f32, b: Point) -> Point {
        Point {
            x: self.x.mul_add(a, b.x),
            y: self.y.mul_add(a, b.y),
        }
    }

    /// Return the Euclidean distance from this point to the origin.
    pub fn hypot(self) -> f32 {
        self.x.hypot(self.y)
    }
}

impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Point {
    type Output = Point;

    fn sub(self, other: Point) -> Point {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<f32> for Point {
    type Output = Point;

    fn mul(self, other: f32) -> Point {
        Point {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl Mul<Point> for Point {
    type Output = f32;

    fn mul(self, other: Point) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl Div<f32> for Point {
    type Output = Point;

    fn div(self, other: f32) -> Point {
        Point {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl Neg for Point {
    type Output = Point;

    fn neg(self) -> Point {
        Point {
            x: -self.x,
            y: -self.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn point_approx_eq(a: Point, b: Point) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y)
    }

    #[test]
    fn add_sums_components() {
        let a = Point::new(1.0, 2.0);
        let b = Point::new(3.0, 4.0);
        assert!(point_approx_eq(a + b, Point::new(4.0, 6.0)));
    }

    #[test]
    fn sub_differences_components() {
        let a = Point::new(5.0, 7.0);
        let b = Point::new(2.0, 3.0);
        assert!(point_approx_eq(a - b, Point::new(3.0, 4.0)));
    }

    #[test]
    fn mul_scalar_scales_both_axes() {
        let p = Point::new(2.0, -3.0);
        assert!(point_approx_eq(p * 2.0, Point::new(4.0, -6.0)));
        assert!(point_approx_eq(p * 0.0, Point::new(0.0, 0.0)));
    }

    #[test]
    fn mul_point_computes_dot_product() {
        let a = Point::new(1.0, 2.0);
        let b = Point::new(3.0, 4.0);
        assert!(approx_eq(a * b, 11.0)); // 1*3 + 2*4
    }

    #[test]
    fn div_scalar_divides_both_axes() {
        let p = Point::new(6.0, -4.0);
        assert!(point_approx_eq(p / 2.0, Point::new(3.0, -2.0)));
    }

    #[test]
    fn neg_flips_sign_of_both_axes() {
        let p = Point::new(3.0, -5.0);
        assert!(point_approx_eq(-p, Point::new(-3.0, 5.0)));
    }

    #[test]
    fn hypot_returns_euclidean_length() {
        let p = Point::new(3.0, 4.0);
        assert!(approx_eq(p.hypot(), 5.0));
        assert!(approx_eq(Point::new(0.0, 0.0).hypot(), 0.0));
    }

    #[test]
    fn mul_add_fused_multiply_add() {
        // result = self * a + b
        let p = Point::new(1.0, 2.0);
        let b = Point::new(10.0, 20.0);
        let result = p.mul_add(3.0, b);
        assert!(point_approx_eq(result, Point::new(13.0, 26.0)));
    }
}
