use nalgebra::{DMatrix, MatrixXx3};

mod edge2;
mod edge3;
mod quad4;
mod quad9;
mod solid8;
mod tria3;
mod tria6;

pub use edge2::Edge2;
pub use edge3::Edge3;
pub use quad4::Quad4;
pub use quad9::Quad9;
pub use solid8::Solid8;
pub use tria3::Tria3;
pub use tria6::Tria6;

use crate::materials::Material;

pub trait GeneralElement {
    fn update(
        &mut self,
        element_number: usize,                   // 单元编号, 即单元的全局索引
        element_node_matrix: &DMatrix<usize>,    // 全局单元-节点编号矩阵
        node_coordinate_matrix: &MatrixXx3<f64>, // 全局节点-坐标矩阵
    );
    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>);
}

/// 用于力学结构计算的单元
/// O(Order)为材料本构矩阵的阶数, 一维O=1, 二维O=3, 三维O=6
pub trait StructureElement<const O: usize> {
    fn structure_calculate(&mut self, mat: &impl Material<O>);
}
