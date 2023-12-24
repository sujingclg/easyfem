mod edge;
pub use edge::*;
mod quad;
pub use quad::*;
mod cube;
pub use cube::*;

mod utils;

use nalgebra::SMatrix;

/// 在单元的每个高斯点上计算形函数、形函数梯度、jacob行列式的结果
/// N -> 单元阶数+1
/// D -> 坐标系维度
pub struct GaussResult<const N: usize, const D: usize> {
    pub shp_val: SMatrix<f64, N, 1>,  // shape function values
    pub shp_grad: SMatrix<f64, N, D>, // shape function gradient matrix
    pub det_j: f64,                   // determinant of jacob matrix
}
