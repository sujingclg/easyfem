mod cube;
mod edge;
mod quad;

pub use cube::Cube8;
pub use edge::{Edge2, Edge3};
pub use quad::{Quad4, Quad9};

use nalgebra::{DMatrix, DVector, MatrixXx3, SMatrix};

use super::utils::flatten_vector;

/// N -> 单元节点个数
/// D -> 坐标系维度
pub trait ElementBase<const N: usize, const D: usize> {
    fn node_dof(&self) -> usize;

    fn connectivity(&self) -> &[usize; N];
    // fn connectivity_mut(&mut self) -> &mut [usize; N];

    fn nodes_coordinates(&self) -> &SMatrix<f64, N, D>;
    // fn nodes_coordinates_mut(&mut self) -> &mut SMatrix<f64, N, D>;

    fn K_mut(&mut self) -> &mut DMatrix<f64>;

    fn F_mut(&mut self) -> &mut DVector<f64>;
}

/// N -> 单元节点个数
/// D -> 坐标系维度
pub trait GeneralElement<const N: usize, const D: usize>: ElementBase<N, D> {
    fn update(
        &mut self,
        element_number: usize,                // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &MatrixXx3<f64>,   // 全局节点-坐标矩阵
    );

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>, right_vector: &mut DVector<f64>) {
        let flattened_connectivity = flatten_vector(self.connectivity(), self.node_dof());
        for (i, node_i) in flattened_connectivity.iter().enumerate() {
            right_vector[*node_i] += self.F_mut()[i];
            for (j, node_j) in flattened_connectivity.iter().enumerate() {
                stiffness_matrix[(*node_i, *node_j)] += self.K_mut()[(i, j)];
            }
        }
        self.K_mut().fill(0.0);
        self.F_mut().fill(0.0);
    }
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
