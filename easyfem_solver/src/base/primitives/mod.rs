mod edge;
pub use edge::*;

mod quad;
pub use quad::*;

mod cube;
pub use cube::*;

use nalgebra::{DMatrix, DVector, MatrixXx3, SMatrix};

use super::utils::flatten_vector;

/// N -> 单元节点个数
/// D -> 坐标系维度
pub trait Primitive<const N: usize, const D: usize> {
    fn node_dof(&self) -> usize;

    fn node_count(&self) -> usize;

    fn connectivity(&self) -> &[usize; N];

    fn nodes_coordinates(&self) -> &SMatrix<f64, N, D>;

    fn K(&self) -> &DMatrix<f64>;
    fn K_mut(&mut self) -> &mut DMatrix<f64>;

    fn F(&self) -> &DVector<f64>;
    fn F_mut(&mut self) -> &mut DVector<f64>;

    /// 从局部 connectivity 的索引拿到全局索引
    fn global_node_id(&self, idx: usize) -> usize {
        self.connectivity()[idx]
    }

    fn update(
        &mut self,
        element_number: usize,                // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &MatrixXx3<f64>,   // 全局节点-坐标矩阵
    );

    fn assemble(&self, stiffness_matrix: &mut DMatrix<f64>, right_vector: &mut DVector<f64>) {
        let flattened_connectivity = flatten_vector(self.connectivity(), self.node_dof());
        for (i, node_i) in flattened_connectivity.iter().enumerate() {
            right_vector[*node_i] += self.F()[i];
            for (j, node_j) in flattened_connectivity.iter().enumerate() {
                stiffness_matrix[(*node_i, *node_j)] += self.K()[(i, j)];
            }
        }
    }

    fn clean(&mut self) {
        self.K_mut().fill(0.0);
        self.F_mut().fill(0.0);
    }
}
