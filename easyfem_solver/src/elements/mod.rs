use nalgebra::{DMatrix, DVector, MatrixXx3, SMatrix};

mod cube8;
mod edge2;
mod edge3;
mod quad4;
mod quad9;

pub use cube8::Cube8;
pub use edge2::Edge2;
pub use edge3::Edge3;
pub use quad4::Quad4;
pub use quad9::Quad9;

use crate::materials::Material;

/// 在单元的每个高斯点上计算形函数、形函数梯度、jacob行列式的结果
/// N -> 单元节点个数
/// D -> 坐标系维度
pub struct GaussResult<const N: usize, const D: usize> {
    pub shape_function_values: SMatrix<f64, N, 1>,
    pub gradient_matrix: SMatrix<f64, N, D>,
    pub det_J: f64,
}

/// N -> 单元节点个数
/// D -> 坐标系维度
pub trait GeneralElement<const N: usize, const D: usize> {
    fn update(
        &mut self,
        element_number: usize,                   // 单元编号, 即单元的全局索引
        element_node_matrix: &DMatrix<usize>,    // 全局单元-节点编号矩阵
        node_coordinate_matrix: &MatrixXx3<f64>, // 全局节点-坐标矩阵
    );

    /// kij: 刚度矩阵算子
    /// fi: 右端项算子
    fn general_stiffness_calculate<K, F>(&mut self, kij_operator: K, fi_operator: F)
    where
        K: Fn(usize, usize, &GaussResult<N, D>) -> DMatrix<f64>,
        F: Fn(usize, &GaussResult<N, D>) -> DVector<f64>;
    // TODO: 尝试用范型处理
    // fn general_stiffness_calculate<K, F, DOF>(&mut self, kij_operator: K, fi_operator: F)
    // where
    //     K: Fn(usize, usize, &GaussResult<N, D>) -> SMatrix<f64, DOF, DOF>,
    //     F: Fn(usize, &GaussResult<N, D>) -> SMatrix<f64, DOF, 1>;

    fn general_assemble(
        &mut self,
        stiffness_matrix: &mut DMatrix<f64>,
        right_vector: &mut DVector<f64>,
    );
}

/// 用于力学结构计算的单元
/// O(Order)为材料本构矩阵的阶数, 一维O=1, 二维O=3, 三维O=6
pub trait StructureElement<const O: usize> {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<O>);

    fn structure_assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>);
}
