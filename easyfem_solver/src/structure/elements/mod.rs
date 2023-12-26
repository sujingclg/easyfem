mod cube;
mod edge;
mod quad;

use crate::{
    base::{elements::ElementBase, gauss::Gauss},
    materials::Material,
};

/// 用于力学结构计算的单元
/// O -> (Order)为材料本构矩阵的阶数, 一维O=1, 二维O=3, 三维O=6
/// N -> 单元节点个数
pub trait StructureElement<const O: usize, const N: usize>: ElementBase<N> {
    fn structure_stiffness_calc(&mut self, gauss: &Gauss, mat: &impl Material<O>);

    // fn structure_assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>);
}

// /// 用于力学结构计算的单元
// /// O(Order)为材料本构矩阵的阶数, 一维O=1, 二维O=3, 三维O=6
// pub trait StructureElement<const O: usize> {
//     fn structure_stiffness_calculate(&mut self, mat: &impl Material<O>);

//     // fn structure_assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>);
// }

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
