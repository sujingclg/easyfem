use gauss_quad::GaussLegendre;
use nalgebra::{MatrixXx2, MatrixXx3, MatrixXx4, RowVector2, RowVector3, RowVector4};

/// 获取1D高斯矩阵, 行数是高斯点个数, 第一列是高斯权重w, 第二列是高斯点坐标xi
pub fn get_gauss_1d_matrix(deg: usize) -> MatrixXx2<f64> {
    let gauss_pount_number = deg;
    let mut gauss_matrix = MatrixXx2::from_element(gauss_pount_number, 0.0);
    // for evevy columns of the above matrix, 0->weight, 1->xi, 2->eta
    let gauss_legendre = GaussLegendre::init(deg);
    let mut n = 0;
    for i in 0..deg {
        gauss_matrix.set_row(
            n,
            &RowVector2::new(gauss_legendre.weights[i], gauss_legendre.nodes[i]),
        );
        n += 1;
    }
    gauss_matrix
}

/// 获取2D高斯矩阵, 行数是高斯点个数, 第一列是高斯权重w, 第二列是高斯点坐标xi, 第三列是高斯点坐标eta
pub fn get_gauss_2d_matrix(deg: usize) -> MatrixXx3<f64> {
    let gauss_pount_number = deg.pow(2);
    let mut gauss_matrix = MatrixXx3::from_element(gauss_pount_number, 0.0);
    // for evevy columns of the above matrix, 0->weight, 1->xi, 2->eta
    let gauss_legendre = GaussLegendre::init(deg);
    let mut n = 0;
    for i in 0..deg {
        for j in 0..deg {
            gauss_matrix.set_row(
                n,
                &RowVector3::new(
                    gauss_legendre.weights[i] * gauss_legendre.weights[j],
                    gauss_legendre.nodes[i],
                    gauss_legendre.nodes[j],
                ),
            );
            n += 1;
        }
    }
    gauss_matrix
}

/// 获取3D高斯矩阵, 行数是高斯点个数, 第一列是高斯权重w, 第二列是高斯点坐标xi, 第三列是高斯点坐标eta, 第四列是zeta
pub fn get_gauss_3d_matrix(deg: usize) -> MatrixXx4<f64> {
    let gauss_pount_number = deg.pow(3);
    let mut gauss_matrix = MatrixXx4::from_element(gauss_pount_number, 0.0);
    // for evevy columns of the above matrix, 0->weight, 1->xi, 2->eta
    let gauss_legendre = GaussLegendre::init(deg);
    let mut n = 0;
    for i in 0..deg {
        for j in 0..deg {
            for k in 0..deg {
                gauss_matrix.set_row(
                    n,
                    &RowVector4::new(
                        gauss_legendre.weights[i]
                            * gauss_legendre.weights[j]
                            * gauss_legendre.weights[k],
                        gauss_legendre.nodes[i],
                        gauss_legendre.nodes[j],
                        gauss_legendre.nodes[k],
                    ),
                );
                n += 1;
            }
        }
    }
    gauss_matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_1d_test() {
        let matrix = get_gauss_1d_matrix(2);
        println!("{}", matrix);
    }

    #[test]
    fn gauss_2d_test() {
        let matrix = get_gauss_2d_matrix(2);
        println!("{}", matrix);
    }

    #[test]
    fn gauss_3d_test() {
        let matrix = get_gauss_3d_matrix(2);
        println!("{}", matrix);
    }
}
