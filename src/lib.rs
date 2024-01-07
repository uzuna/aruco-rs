use image::{ImageBuffer, Luma};
use imageproc::{contours::Contour, point};
use nalgebra::Normed;

#[derive(Debug, Default)]
struct Poly {
    points: Vec<nalgebra::Point2<f32>>,
}

// fn find_candidate(
//     cs: &[Contour<u32>],
//     min_size: usize,
//     epsilon: f32,
//     min_length: usize,
// ) -> Vec<Poly> {
//     for c in cs {
//         if c.points.len() >= min_size {
//             let c = imageproc::contours::find_contours_with_threshold::<u32>(&img, 4);
//             for i in c {
//                 for p in i.points {
//                     dst.put_pixel(p.x, p.y, Luma::from([128u8]));
//                 }
//                 println!("{:?}", i.border_type);
//             }
//         }
//     }
//     vec![]
// }

#[derive(Debug, Clone, Copy)]
struct Candidate {
    point: nalgebra::Point2<f32>,
    index: usize,
}

impl Candidate {
    fn from_points(points: &[nalgebra::Point2<f32>], index: usize) -> Self {
        Self {
            point: points[index],
            index,
        }
    }
}

// 閉じた線形
//
// 輪郭検出の結果なので何らかの閉じた線形で、その線分を示す点で構成される
// n個野天がある場合n+1は0番目の点と同じになる
struct ClosedShape(Vec<nalgebra::Point2<f32>>);

impl ClosedShape {
    fn from_contour(c: &Contour<u32>) -> Self {
        Self(
            c.points
                .iter()
                .map(|p| nalgebra::Point2::new(p.x as f32, p.y as f32))
                .collect(),
        )
    }
}

// pointsの中の二点の距離
#[derive(Debug)]
struct SubPath {
    left: usize,
    right: usize,
}

impl SubPath {
    fn new(left: usize, right: usize) -> Self {
        Self { left, right }
    }
    // 始点と終点を除いた点のイテレータを返す
    fn iter(&self, len: usize) -> SubPathIter {
        SubPathIter {
            len,
            end: self.right,
            current: self.left + 1,
        }
    }
    fn is_next(&self, len: usize) -> bool {
        let index = self.left + 1 % len;
        index == self.right
    }
    // 2点間に指定距離以上の垂線を持つ点があれば分割する
    fn line_or_split(&self, points: &[nalgebra::Point2<f32>], epsilon: f32) -> Option<[Self; 2]> {
        if self.is_next(points.len()) {
            return None;
        }
        // distは線分
        let mut max_dist = 0.0;
        let mut max_index = 0;
        let start = points[self.left];
        let end = points[self.right];

        for i in self.left + 1..self.right {
            let dist = point_line_distance(&points[i], &start, &end);
            if dist > max_dist {
                max_dist = dist;
                max_index = i;
            }
        }
        if max_dist > epsilon.powi(2) * (end-start).norm() {
            Some([
                Self::new(self.left, max_index),
                Self::new(max_index, self.right),
            ])
        } else {
            None
        }
    }
}

struct SubPathIter {
    len: usize,
    end: usize,
    current: usize,
}

impl Iterator for SubPathIter {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        let ret = self.current;
        self.current = (self.current + 1) % self.len;
        if self.current == self.end {
            None
        } else {
            Some(ret)
        }
    }
}

// 直線から点までの距離の・ようなものを計算する
//
// ここでは直線上の左側と右側の面積の違いを直線からの距離とする。
// https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
// TODO 距離のようなものの数学的根拠を調べる
fn point_line_distance(
    p: &nalgebra::Point2<f32>,
    start: &nalgebra::Point2<f32>,
    end: &nalgebra::Point2<f32>,
) -> f32 {
    let v = end - start;
    let w = p - start;
    perpendicular_dist(v, w)
}

#[inline]
fn perpendicular_dist(
    base: nalgebra::Vector2<f32>,
    target: nalgebra::Vector2<f32>,
) -> f32 {
    (target.y * base.x - target.x * base.y).abs()
}


// 輪郭近似
fn approx_poly_dp(c: &Contour<u32>, epsilon: f32) -> Poly {
    let epsilon = epsilon.powi(2);

    let points = ClosedShape::from_contour(c);
    let start_pt = Candidate::from_points(&points.0, 0);
    // start_ptから最も遠い点を探す
    let (max_dist, far_pt) = {
        let mut max_dist = 0.0;
        let mut far_pt = start_pt;
        for (i, p) in points.0.iter().enumerate() {
            let dist = nalgebra::distance_squared(&start_pt.point, p);
            if dist > max_dist {
                max_dist = dist;
                far_pt = Candidate::from_points(&points.0, i);
            }
        }
        (max_dist, far_pt)
    };
    if max_dist < epsilon {
        return Poly {
            points: vec![start_pt.point],
        };
    }

    // 2点のパスの間を指定の誤差の誤差の点を省略する形で点数を削減
    // 始点と終点の直線状から最も遠い点を探し、誤差範囲外ならパスを分割する
    let mut stack = vec![
        SubPath::new(start_pt.index, far_pt.index),
        SubPath::new(far_pt.index, start_pt.index),
    ];
    let mut poly = Poly::default();

    while !stack.is_empty() {
        let path = stack.pop().unwrap();
        if let Some([left, right]) = path.line_or_split(&points.0, epsilon) {
            stack.push(left);
            stack.push(right);
        } else {
            poly.points.push(points.0[path.left]);
            poly.points.push(points.0[path.right]);
        }
    }
    poly
}

// 凸包と仮定して、最も遠い点の組み合わせを探す
fn find_convexity_far_point(points: &[nalgebra::Point2<f32>]) -> SubPath {
    let mut max_dist = 0.0;
    let mut max_index = 0;
    let mut start_index = 0;
    for _ in 0..3 {
        start_index = max_index;
        max_dist = 0.0;
        max_index = 0;
        for i in 0..points.len() {
            let dist = nalgebra::distance_squared(&points[start_index], &points[i]);
            if dist > max_dist {
                max_dist = dist;
                max_index = i;
            }
        }
    }
    if start_index > max_index {
        SubPath::new(max_index, start_index)
    } else {
        SubPath::new(start_index, max_index)
    }
}

fn line_length(points: &[nalgebra::Point2<f32>]) -> f32 {
    let mut length = 0.0;
    for i in 0..points.len() {
        let next = (i + 1) % points.len();
        length += nalgebra::distance(&points[i], &points[next]);
    }
    length
}

fn approx_line(points: &[nalgebra::Point2<f32>], epsilon: f32) -> Poly {
    let far = find_convexity_far_point(points);
    let mut stack = vec![far];
    let mut poly = Poly::default();

    let path = stack.get(0).unwrap();
    poly.points.push(points[path.left]);
    while !stack.is_empty() {
        let path = stack.pop().unwrap();
        if let Some([left, right]) = path.line_or_split(points, epsilon) {
            stack.push(right);
            stack.push(left);
        } else {
            poly.points.push(points[path.right]);
        }
    }
    poly
}

#[cfg(test)]
mod tests {
    use image::{ImageBuffer, Luma};

    use crate::approx_poly_dp;

    #[ignore]
    #[test]
    fn it_works() {
        let img = image::open("testdata/test.jpeg").unwrap();
        let img = img.to_luma8();
        // reduce pixel noise
        let img = imageproc::filter::box_filter(&img, 3, 3);
        // to binary image
        let otsu_level = imageproc::contrast::otsu_level(&img);
        let img = imageproc::contrast::threshold(&img, otsu_level);
        // 輪郭検出
        let mut dst = ImageBuffer::new(img.width(), img.height());
        let c = imageproc::contours::find_contours_with_threshold::<u32>(&img, 4);
        for i in c {
            // 候補探索
            let poly = approx_poly_dp(&i, 10.0);
            for p in i.points {
                dst.put_pixel(p.x, p.y, Luma::from([128u8]));
            }
            for p in poly.points {
                dst.put_pixel(p.x as u32, p.y as u32, Luma::from([255u8]));
            }
        }
        // 候補の回転
        // 大きさに差がある場合に。近くのものは削除?

        // 画像の保存
        img.save_with_format("testdata/out.png", image::ImageFormat::Png)
            .unwrap();
        dst.save_with_format("testdata/dst.png", image::ImageFormat::Png)
            .unwrap();
    }

    #[test]
    fn test_distance_from_line() {
        let start = nalgebra::Point2::new(0.0, 0.0);
        let end = nalgebra::Point2::new(2.0, 2.0);
        #[derive(Debug)]
        struct TestCase {
            point: nalgebra::Point2<f32>,
            dist: f32,
        }
        let testdata = vec![
            TestCase {
                point: nalgebra::Point2::new(0.5, 0.5),
                dist: 0.0,
            },
            TestCase {
                point: nalgebra::Point2::new(1.0, 1.0),
                dist: 0.0,
            },
            TestCase {
                point: nalgebra::Point2::new(1.0, 0.0),
                dist:2.0,
            },
            TestCase {
                point: nalgebra::Point2::new(0.0, 1.0),
                dist:2.0,
            },
            TestCase {
                point: nalgebra::Point2::new(2.0, 1.0),
                dist:2.0,
            },
            TestCase {
                point: nalgebra::Point2::new(1.0, 2.0),
                dist:2.0,
            },
            TestCase {
                point: nalgebra::Point2::new(0.0, 2.0),
                dist: 4.0,
            },
        ];
        for t in testdata {
            let dist = super::point_line_distance(&t.point, &start, &end);
            assert_eq!(dist, t.dist);
        }
        let start = nalgebra::Point2::new(1.0, 0.0);
        let end = nalgebra::Point2::new(2.0, 2.0);
        let testdata = vec![
            TestCase {
                point: nalgebra::Point2::new(1.5, 1.0),
                dist: 0.0,
            },
            TestCase {
                point: nalgebra::Point2::new(1.0, 1.0),
                dist: 1.0,
            },
            TestCase {
                point: nalgebra::Point2::new(1.0, 2.0),
                dist: 2.0,
            },
        ];
        for t in testdata {
            let dist = super::point_line_distance(&t.point, &start, &end);
            assert_eq!(dist, t.dist, "{:?}", t);
        }
    }

    #[test]
    fn test_applox_line() {
        // テストデータ作成 Sの字
        let points = vec![
            nalgebra::Point2::new(0.0, 0.0),
            nalgebra::Point2::new(1.0, 0.0),
            nalgebra::Point2::new(1.0, 1.0),
            nalgebra::Point2::new(1.0, 2.0),
            nalgebra::Point2::new(2.0, 2.0),
        ];
        // 凸包を作って、最も遠い2点を探す -> 輪郭になっているので3回最遠点を取ることで確率的に近似できるとする
        let far = super::find_convexity_far_point(&points);
        assert_eq!(far.left, 0);
        assert_eq!(far.right, 4);
        // 2点の間を一定の誤差で結ぶ直線で近似する
        // 結果はSの真ん中の点だけ省略される
        let poly = super::approx_line(&points, 0.1);
        assert_eq!(poly.points.len(), 4);
        assert_eq!(poly.points[0], points[0]);
        assert_eq!(poly.points[1], points[1]);
        assert_eq!(poly.points[2], points[3]);
        assert_eq!(poly.points[3], points[4]);

        // 2点の間を一定の誤差で結ぶ直線で近似する ep = 1.5なので
        // 2点間の距離の1.0倍以内の点は省略される
        let poly = super::approx_line(&points, 1.0);
        assert_eq!(poly.points.len(), 2);
    }
}
