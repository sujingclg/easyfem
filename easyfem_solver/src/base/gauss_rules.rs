use gauss_quad::GaussLegendre;
use nalgebra::{MatrixXx3, MatrixXx4, RowVector3, RowVector4};

//     point_number: usize,          // 高斯点个数
//     point_coords: MatrixXx2<f64>, // 0-> weight, 1-> xi, 行数为高斯点个数
// }

// impl GaussPoint1D {
//     pub fn new(point_number: usize) -> Self {
//         // let point_coords = MatrixXx2::from_element(point_number, 0.0);
//         // let y: Matrix<f64, Const<point_number>, Const<2>, VecStorage<f64, Dyn, U2>> =
//         //     Matrix2::from_element(0.0);
//         let point_coords = match point_number {
//             1 => MatrixXx2::from_rows(&[RowVector2::new(2.0, 0.0)]),
//             2 => MatrixXx2::from_rows(&[
//                 RowVector2::new(1.0, f64::sqrt(1.0 / 3.0)),
//                 RowVector2::new(1.0, f64::sqrt(1.0 / 3.0)),
//             ]),
//             3 => MatrixXx2::from_rows(&[
//                 RowVector2::new(5.0 / 9.0, f64::sqrt(3.0 / 5.0)),
//                 RowVector2::new(8.0 / 9.0, 0.0),
//                 RowVector2::new(5.0 / 9.0, f64::sqrt(3.0 / 5.0)),
//             ]),
//             // 4 => MatrixXx2::from_rows(&[
//             //     RowVector2::new(5.0 / 9.0, f64::sqrt(3.0 / 5.0)),
//             //     RowVector2::new(8.0 / 9.0, 0.0),
//             //     RowVector2::new(5.0 / 9.0, f64::sqrt(3.0 / 5.0)),
//             // ]),
//             // 5 => MatrixXx2::from_rows(&[
//             //     RowVector2::new(5.0 / 9.0, f64::sqrt(3.0 / 5.0)),
//             //     RowVector2::new(8.0 / 9.0, 0.0),
//             //     RowVector2::new(5.0 / 9.0, f64::sqrt(3.0 / 5.0)),
//             // ]),
//             _ => {
//                 panic!("point_number must bigger than 0 and letter than 4");
//                 // MatrixXx2::from_element(point_number, 0.0)
//             }
//         };
//         println!("{}", point_coords);
//         GaussPoint1D {
//             point_number,
//             point_coords,
//         }
//     }

//     fn create_gauss_points(&self) {}
// }

// pub struct GaussPoint2D {
//     point_number: usize,          // 高斯点个数
//     point_coords: MatrixXx3<f64>, // 0->weight,1->xi,2->eta, 行数为高斯点个数
// }

// impl GaussPoint2D {
//     pub fn new(point_number: usize) -> Self {
//         GaussPoint2D {
//             point_number,
//             point_coords: MatrixXx3::from_element(point_number, 0.0),
//         }
//     }
// }

pub fn gauss_1d_integrate<F>(deg: usize, integrand: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let gauss_legendre = GaussLegendre::init(deg);
    gauss_legendre.integrate(-1.0, 1.0, integrand)
}

pub fn gauss_2d_integrate<F>(deg: usize, integrand: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let gauss_pount_number = deg.pow(2);
    let mut gauss_point_coords = MatrixXx3::from_element(gauss_pount_number, 0.0);
    // for evevy columns of the above matrix, 0->weight, 1->xi, 2->eta
    let gauss_legendre = GaussLegendre::init(deg);
    let mut k = 0;
    for i in 0..deg {
        for j in 0..deg {
            gauss_point_coords.set_row(
                k,
                &RowVector3::new(
                    gauss_legendre.weights[i] * gauss_legendre.weights[j],
                    gauss_legendre.nodes[i],
                    gauss_legendre.nodes[j],
                ),
            );
            k += 1;
        }
    }
    // println!("{}", gauss_point_coords);
    let mut sum = 0.0;
    for i in 0..gauss_pount_number {
        let (x, y, w) = (
            gauss_point_coords[(i, 1)],
            gauss_point_coords[(i, 2)],
            gauss_point_coords[(i, 0)],
        );
        sum += integrand(x, y) * w;
    }

    // let c0 = DVector::from_column_slice(&gauss_legendre.weights);
    // let c1 = DVector::from_column_slice(&gauss_legendre.nodes);
    // let c2 = DVector::from_column_slice(&gauss_legendre.nodes);
    // let x = MatrixXx3::from_columns(&[c0, c1, c2]);
    sum
}

pub fn gauss_3d_integrate<F>(deg: usize, integrand: F) -> f64
where
    F: Fn(f64, f64, f64) -> f64,
{
    let gauss_pount_number = deg.pow(3);
    let mut gauss_point_coords = MatrixXx4::from_element(gauss_pount_number, 0.0);
    // for evevy columns of the above matrix, 0->weight, 1->xi, 2->eta, 3-> zeta
    let gauss_legendre = GaussLegendre::init(deg);
    let mut l = 0;
    for i in 0..deg {
        for j in 0..deg {
            for k in 0..deg {
                gauss_point_coords.set_row(
                    l,
                    &RowVector4::new(
                        gauss_legendre.weights[i]
                            * gauss_legendre.weights[j]
                            * gauss_legendre.weights[k],
                        gauss_legendre.nodes[i],
                        gauss_legendre.nodes[j],
                        gauss_legendre.nodes[k],
                    ),
                );
                l += 1;
            }
        }
    }
    // println!("{}", gauss_point_coords);
    let mut sum = 0.0;
    for i in 0..gauss_pount_number {
        let (x, y, z, w) = (
            gauss_point_coords[(i, 1)],
            gauss_point_coords[(i, 2)],
            gauss_point_coords[(i, 3)],
            gauss_point_coords[(i, 0)],
        );
        sum += integrand(x, y, z) * w;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d() {
        let real_result = f64::sin(1.0) - f64::sin(-1.0);
        println!("real_result: {:?}", real_result);
        let mut err = 100.0;
        for i in 1..5 {
            let integral = gauss_1d_integrate(i, |x| f64::cos(x));
            err = f64::abs(real_result - integral) / real_result;
            println!("{}%", err * 100.0);
        }
        assert!(
            err < 1e-6,
            "Calculation accuracy does not meet requirements."
        );
    }

    #[test]
    fn test_2d() {
        let real_result = 4.0 / 15.0;
        let mut err = 100.0;
        for i in 1..6 {
            let integral = gauss_2d_integrate(i, |x, y| x.powi(2) * y.powi(4));
            err = f64::abs(real_result - integral) / real_result;
            println!("{}%", err * 100.0);
        }
        assert!(
            err < 1e-6,
            "Calculation accuracy does not meet requirements."
        );
    }

    #[test]
    fn test_3d() {
        let real_result = f64::powi(2.0 / 3.0, 3);
        let mut err = 100.0;
        for i in 1..4 {
            let integral = gauss_3d_integrate(i, |x, y, z| x.powi(2) * y.powi(2) * z.powi(2));
            err = f64::abs(real_result - integral) / real_result;
            println!("{}%", err * 100.0);
        }
        assert!(
            err < 1e-6,
            "Calculation accuracy does not meet requirements."
        );
    }
}
