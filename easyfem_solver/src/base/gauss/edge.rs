use nalgebra::{Matrix2x1, Matrix3x1, Matrix4x1, SMatrix, Vector2, Vector3, Vector4};

use super::{
    utils::{gauss_1d_vector, jacob_determinant},
    Gauss, GaussResult,
};

/// N -> 单元节点个数
pub struct GaussEdge<const N: usize> {
    gauss_vector: Vec<(f64, [f64; 1])>, // 高斯积分矩阵, 行数为高斯积分点个数
                                        // 1列->w 2列->xi
}

impl<const N: usize> GaussEdge<N> {
    pub fn new(gauss_deg: usize) -> Self {
        GaussEdge {
            gauss_vector: gauss_1d_vector(gauss_deg),
        }
    }
}

pub type GaussEdge2 = GaussEdge<2>;

impl Gauss<2, 1> for GaussEdge2 {
    fn gauss_vector(&self) -> &Vec<(f64, [f64; 1])> {
        &self.gauss_vector
    }

    fn shape_func_calc(
        &self,
        nodes_coordinates: &SMatrix<f64, 2, 1>, // 单元的节点坐标矩阵
        gauss_point: &[f64; 1],
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

        let det_j = jacob_determinant(&shp_grad, &nodes_coordinates);
        shp_grad /= det_j; // 梯度矩阵除以jacob行列式以便进行单元组装

        GaussResult {
            shp_val,
            shp_grad,
            det_j,
        }
    }
}

pub type GaussEdge3 = GaussEdge<3>;

impl Gauss<3, 1> for GaussEdge3 {
    fn gauss_vector(&self) -> &Vec<(f64, [f64; 1])> {
        &self.gauss_vector
    }

    fn shape_func_calc(
        &self,
        nodes_coordinates: &SMatrix<f64, 3, 1>, // 单元的节点坐标矩阵
        gauss_point: &[f64; 1],
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

        let det_j = jacob_determinant(&shp_grad, &nodes_coordinates);
        shp_grad /= det_j; // 梯度矩阵除以jacob行列式以便进行单元组装

        GaussResult {
            shp_val,
            shp_grad,
            det_j,
        }
    }
}

pub type GaussEdge4 = GaussEdge<4>;

impl Gauss<4, 1> for GaussEdge4 {
    fn gauss_vector(&self) -> &Vec<(f64, [f64; 1])> {
        &self.gauss_vector
    }

    fn shape_func_calc(
        &self,
        nodes_coordinates: &SMatrix<f64, 4, 1>, // 单元的节点坐标矩阵
        gauss_point: &[f64; 1],
    ) -> GaussResult<4, 1> {
        let [xi] = gauss_point;

        // TODO: 代码是直接cv过来的, 需要后续再研究一下
        // 1维4节点等参元形函数
        let shp_val = Vector4::new(
            -(3.0 * xi + 1.0) * (3.0 * xi - 1.0) * (xi - 1.0) / 16.0, // N1
            (xi + 1.0) * (3.0 * xi + 1.0) * (3.0 * xi - 1.0) / 16.0,  // N2
            (3.0 * xi + 3.0) * (3.0 * xi - 1.0) * (3.0 * xi - 3.0) / 16.0, // N3
            -(3.0 * xi + 3.0) * (3.0 * xi + 1.0) * (3.0 * xi - 3.0) / 16.0, // N4
        );

        // 梯度矩阵，每个元素分别是形函数对xi求偏导: \frac{\partial Ni}{\partial \xi}
        let mut shp_grad = Matrix4x1::new(
            -27.0 * xi * xi / 16.0 + 9.0 * xi / 8.0 + 1.0 / 16.0, // dN1/dxi
            27.0 * xi * xi / 16.0 + 9.0 * xi / 8.0 - 1.0 / 16.0,  // dN2/dxi
            81.0 * xi * xi / 16.0 - 9.0 * xi / 8.0 - 27.0 / 16.0, // dN3/dxi
            -81.0 * xi * xi / 16.0 - 9.0 * xi / 8.0 + 27.0 / 16.0, // dN4/dxi
        );

        let det_j = jacob_determinant(&shp_grad, &nodes_coordinates);
        shp_grad /= det_j;

        GaussResult {
            shp_val,
            shp_grad,
            det_j,
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn test() {}
}
