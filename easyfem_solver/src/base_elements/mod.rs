mod cube;
mod edge;
mod quad;

pub use cube::Cube8;
pub use edge::{Edge2, Edge3};
pub use quad::{Quad4, Quad9};

use nalgebra::{DMatrix, DVector, MatrixXx3};

use crate::materials::Material;

pub trait GeneralElement {
    fn update(
        &mut self,
        element_number: usize,                // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &MatrixXx3<f64>,   // 全局节点-坐标矩阵
    );

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>, right_vector: &mut DVector<f64>);
}

/// 用于力学结构计算的单元
/// O(Order)为材料本构矩阵的阶数, 一维O=1, 二维O=3, 三维O=6
pub trait StructureElement<const O: usize> {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<O>);

    // fn structure_assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>);
}

// impl<T, const O: usize> StructureElement<O> for T
// where
//     T: GeneralElement,
// {
//     fn structure_assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>) {
//         let mut right_vector = DVector::zeros(stiffness_matrix.nrows());
//         self.assemble(stiffness_matrix, &mut right_vector);
//     }

//     fn structure_stiffness_calculate(&mut self, mat: &impl Material<O>) {}
// }
