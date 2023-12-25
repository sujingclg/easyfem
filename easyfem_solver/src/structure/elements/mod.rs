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
/// D -> 坐标系维度
pub trait StructureElement<const O: usize, const N: usize, const D: usize>:
    ElementBase<N, D>
{
    fn structure_stiffness_calc(&mut self, gauss: &Gauss, mat: &impl Material<O>);

    // fn structure_assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>);
}
