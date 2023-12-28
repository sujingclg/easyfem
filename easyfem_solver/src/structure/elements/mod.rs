mod edge;
pub use edge::*;

mod quad;
pub use quad::*;

mod cube;
pub use cube::*;

use nalgebra::{DMatrix, DVector, MatrixXx3};

use crate::materials::Material;

/// 用于力学结构计算的单元
/// O -> (Order)为材料本构矩阵的阶数, 一维O=1, 二维O=3, 三维O=6
/// N -> 单元节点个数
/// D -> 坐标系维度
pub trait StructureElement<const O: usize> {
    fn update(
        &mut self,
        element_number: usize,                // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &MatrixXx3<f64>,   // 全局节点-坐标矩阵
    );

    fn structure_stiffness_calc(&mut self, mat: &impl Material<O>);

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>, right_vector: &mut DVector<f64>);
}
