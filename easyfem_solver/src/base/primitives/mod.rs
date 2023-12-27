mod edge;
pub use edge::*;

mod quad;
pub use quad::*;

mod cube;
pub use cube::*;

use nalgebra::{DMatrix, DVector, MatrixXx3};

use super::utils::flatten_vector;

/// N -> 单元节点个数
pub trait PrimitiveBase<const N: usize> {
    type CoordMatrix;

    fn node_dof(&self) -> usize;

    fn node_count(&self) -> usize;

    fn connectivity(&self) -> &[usize; N];
    // fn connectivity_mut(&mut self) -> &mut [usize; N];

    fn nodes_coordinates(&self) -> &Self::CoordMatrix;
    // fn nodes_coordinates_mut(&mut self) -> &mut SMatrix<f64, N, D>;

    fn K_mut(&mut self) -> &mut DMatrix<f64>;

    fn F_mut(&mut self) -> &mut DVector<f64>;
}

/// N -> 单元节点个数
pub trait GeneralElement<const N: usize>: PrimitiveBase<N> {
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
