use nalgebra::{Matrix4x2, MatrixXx3, SMatrix, Vector4};

use super::{utils::gauss_2d_matrix, GaussResult};

pub struct GaussQuad {
    gauss_matrix: MatrixXx3<f64>, // 高斯积分矩阵, 行数为高斯积分点个数
                                  // 1列->w 2列->xi 3列->eta
}

impl GaussQuad {
    pub fn new(gauss_deg: usize) -> Self {
        GaussQuad {
            gauss_matrix: gauss_2d_matrix(gauss_deg),
        }
    }

    pub fn gauss_matrix(&self) -> &MatrixXx3<f64> {
        &self.gauss_matrix
    }

    pub fn linear_shape_func_calc(
        &self,
        nodes_coordinates: &SMatrix<f64, 4, 2>,
        gauss_point: [f64; 2],
    ) -> GaussResult<4, 2> {
        let [xi, eta] = gauss_point;
        // 2维4节点等参元形函数
        let shp_val = Vector4::new(
            (1.0 - xi) * (1.0 - eta) / 4.0, // N1
            (1.0 + xi) * (1.0 - eta) / 4.0, // N2
            (1.0 + xi) * (1.0 + eta) / 4.0, // N3
            (1.0 - xi) * (1.0 + eta) / 4.0, // N4
        );

        // 梯度矩阵，每个元素分别是形函数分别对xi和eta求偏导: \frac{\partial Ni}{\partial \xi} \frac{\partial Ni}{\partial \eta}
        let mut shp_grad = Matrix4x2::new(
            (eta - 1.0) / 4.0,  // dN1/dxi
            (xi - 1.0) / 4.0,   // dN1/deta
            (1.0 - eta) / 4.0,  // dN2/dxi
            -(1.0 + xi) / 4.0,  // dN2/deta
            (1.0 + eta) / 4.0,  // dN3/dxi
            (1.0 + xi) / 4.0,   // dN3/deta
            -(1.0 + eta) / 4.0, // dN4/xi
            (1.0 - xi) / 4.0,   // dN4/deta
        );

        // jacob矩阵形式为[[dx_dxi, dy_dxi],[dx_deta, dy_deta]]
        let jacob_matrix = shp_grad.transpose() * nodes_coordinates;
        let det_j = jacob_matrix.determinant(); // 行列式

        // jacob逆矩阵形式为[[dxi_dx, deta_dx],[dxi_dy, deta_dy]]
        let inverse_j = jacob_matrix.try_inverse().unwrap(); // 逆矩阵

        // shp_grad 的每一行变为下式, Ni从N1到N4
        // dNi/dxi * dxi/dx + dNi/deta * deta/dx, dNi/dxi * dxi/dy + dNi/deta * deta/dy
        shp_grad = shp_grad * inverse_j.transpose();

        GaussResult {
            shp_val,
            shp_grad,
            det_j,
        }
    }

    pub fn square_shape_func_calc(
        &self,
        nodes_coordinates: &SMatrix<f64, 9, 2>,
        gauss_point: [f64; 2],
    ) -> GaussResult<9, 2> {
        let [xi, eta] = gauss_point;
        // 2维9节点等参元形函数
        let shp_val = SMatrix::from_column_slice(&[
            (xi * xi - xi) * (eta * eta - eta) / 4.0,  // N1
            (xi * xi + xi) * (eta * eta - eta) / 4.0,  // N2
            (xi * xi + xi) * (eta * eta + eta) / 4.0,  // N3
            (xi * xi - xi) * (eta * eta + eta) / 4.0,  // N4
            (1.0 - xi * xi) * (eta * eta - eta) / 2.0, // N5
            (xi * xi + xi) * (1.0 - eta * eta) / 2.0,  // N6
            (1.0 - xi * xi) * (eta * eta + eta) / 2.0, // N7
            (xi * xi - xi) * (1.0 - eta * eta) / 2.0,  // N8
            (1.0 - xi * xi) * (1.0 - eta * eta),       // N9
        ]);

        let mut shp_grad = SMatrix::<f64, 9, 2>::from_row_slice(&[
            (2.0 * xi - 1.0) * (eta * eta - eta) / 4.0, // dN1/dxi
            (xi * xi - xi) * (2.0 * eta - 1.0) / 4.0,   // dN1/deta
            //
            (2.0 * xi + 1.0) * (eta * eta - eta) / 4.0, // dN2/dxi
            (xi * xi + xi) * (2.0 * eta - 1.0) / 4.0,   // dN2/deta
            //
            (2.0 * xi + 1.0) * (eta * eta + eta) / 4.0, // dN3/dxi
            (xi * xi + xi) * (2.0 * eta + 1.0) / 4.0,   // dN3/deta
            //
            (2.0 * xi - 1.0) * (eta * eta + eta) / 4.0, // dN4/dxi
            (xi * xi - xi) * (2.0 * eta + 1.0) / 4.0,   // dN4/deta
            //
            -xi * (eta * eta - eta),                   // dN5/dxi
            (1.0 - xi * xi) * (2.0 * eta - 1.0) / 2.0, // dN5/deta
            //
            (2.0 * xi + 1.0) * (1.0 - eta * eta) / 2.0, // dN6/dxi
            -(xi * xi + xi) * eta,                      // dN6/deta
            //
            -xi * (eta * eta + eta),                   // dN7/dxi
            (1.0 - xi * xi) * (2.0 * eta + 1.0) / 2.0, // dN7/deta
            //
            (2.0 * xi - 1.0) * (1.0 - eta * eta) / 2.0, // dN8/dxi
            -(xi * xi - xi) * eta,                      // dN8/deta
            //
            -2.0 * xi * (1.0 - eta * eta), // dN9/dxi
            -2.0 * eta * (1.0 - xi * xi),  // dN9/deta
        ]);

        let jacob_matrix = shp_grad.transpose() * nodes_coordinates;
        let det_j = jacob_matrix.determinant();

        let inverse_j = jacob_matrix.try_inverse().unwrap();

        shp_grad = shp_grad * inverse_j.transpose();

        GaussResult {
            shp_val,
            shp_grad,
            det_j,
        }
    }

    // pub fn cubic_shape_func_calc() {}
}
