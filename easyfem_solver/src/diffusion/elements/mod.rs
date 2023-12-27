mod edge;
pub use edge::*;

mod quad;

use nalgebra::{DMatrix, DVector, MatrixXx3};

/// N -> 单元节点个数
/// D -> 坐标系维度
pub trait DiffusionElement {
    fn update(
        &mut self,
        element_number: usize,                // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &MatrixXx3<f64>,   // 全局节点-坐标矩阵
    );

    fn diffusion_stiffness_calc(&mut self, diffusivity: f64, dt: f64, prev_solution: &DVector<f64>);

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>, right_vector: &mut DVector<f64>);
}
