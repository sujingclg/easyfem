mod edge;
pub use edge::{GaussEdge2, GaussEdge3, GaussEdge4};

mod quad;
pub use quad::{GaussQuad4, GaussQuad9};

mod cube;
pub use cube::GaussCube8;

mod utils;

use nalgebra::SMatrix;

/// 在单元的每个高斯点上计算形函数、形函数梯度、jacob行列式的结果
/// N -> 单元节点数
/// D -> 坐标系维度
pub struct GaussResult<const N: usize, const D: usize> {
    pub shp_val: SMatrix<f64, N, 1>,  // shape function values
    pub shp_grad: SMatrix<f64, N, D>, // shape function gradient matrix
    pub det_j: f64,                   // determinant of jacob matrix
}

pub trait Gauss<const N: usize, const D: usize> {
    fn gauss_vector(&self) -> &Vec<(f64, [f64; D])>; // (weight, [gauss_point])

    fn shape_func_calc(
        &self,
        nodes_coordinates: &SMatrix<f64, N, D>, // 单元的节点坐标矩阵
        gauss_point: &[f64; D],
    ) -> GaussResult<N, D>;
}
