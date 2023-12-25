use nalgebra::{Matrix2x1, Matrix3x1, MatrixXx2, SMatrix, Vector2, Vector3};

use super::{utils::gauss_1d_matrix, GaussResult};

pub struct GaussEdge {
    gauss_matrix: MatrixXx2<f64>, // 高斯积分矩阵, 行数为高斯积分点个数
                                  // 1列->w 2列->xi
}

impl GaussEdge {
    pub fn new(gauss_deg: usize) -> Self {
        GaussEdge {
            gauss_matrix: gauss_1d_matrix(gauss_deg),
        }
    }

    pub fn gauss_matrix(&self) -> &MatrixXx2<f64> {
        &self.gauss_matrix
    }

    pub fn linear_shape_func_calc(
        &self,
        nodes_coordinates: &SMatrix<f64, 2, 1>, // 单元的节点坐标矩阵
        gauss_point: [f64; 1],
    ) -> GaussResult<2, 1> {
        let [xi] = gauss_point;
        // 1维2节点等参元形函数
        let shp_val = Vector2::new(
            (1.0 - xi) / 2.0, // N1
            (1.0 + xi) / 2.0, // N2
        );

        // 梯度矩阵，每个元素分别是形函数对xi求偏导: \frac{\partial Ni}{\partial \xi}
        let mut shp_grad = Matrix2x1::new(
            -0.5, // dN1/dxi
            0.5,  // dN2/dxi
        );

        let mut dx_dxi = 0.0;

        // 遍历单元的每个节点
        for i in 0..2 {
            dx_dxi += shp_grad[i] * nodes_coordinates[i];
        }
        // let det_jacob = f64::abs(dx_dxi);
        let det_j = dx_dxi; // jacob行列式

        shp_grad /= det_j; // 梯度矩阵除以jacob行列式以便进行单元组装

        GaussResult {
            shp_val,
            shp_grad,
            det_j,
        }
    }

    pub fn square_shape_func_calc(
        &self,
        nodes_coordinates: &SMatrix<f64, 3, 1>, // 单元的节点坐标矩阵
        gauss_point: [f64; 1],
    ) -> GaussResult<3, 1> {
        let [xi] = gauss_point;
        // 1维3节点等参元形函数
        let shp_val = Vector3::new(
            xi * (xi - 1.0) * 0.5, // N1
            xi * (xi + 1.0) * 0.5, // N2
            1.0 - xi.powi(2),      // N3
        );

        // 梯度矩阵，每个元素分别是形函数对xi求偏导: \frac{\partial Ni}{\partial \xi}
        let mut shp_grad = Matrix3x1::new(
            xi - 0.5,  // dN1/dxi
            xi + 0.5,  // dN2/dxi
            -2.0 * xi, // dN3/dxi
        );

        let mut dx_dxi = 0.0;

        // 遍历单元的每个节点
        for i in 0..3 {
            dx_dxi += shp_grad[i] * nodes_coordinates[i];
        }
        // let det_J = f64::abs(dx_dxi);
        let det_j = dx_dxi; // jacob行列式

        shp_grad /= det_j; // 梯度矩阵除以jacob行列式以便进行单元组装

        GaussResult {
            shp_val,
            shp_grad,
            det_j,
        }
    }

    // pub fn cubic_shape_func_calc() {}
}
