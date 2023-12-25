use nalgebra::{DMatrix, DVector, MatrixXx3, SMatrix};

use super::{ElementBase, GeneralElement};

pub type Cube8 = Cube<8>;

pub struct Cube<const N: usize> {
    node_dof: usize,                       // 节点自由度 结构分析为3
    connectivity: [usize; N],              // 单元的节点序号数组
    nodes_coordinates: SMatrix<f64, N, 3>, // 单元节点的全局坐标数组, 每单元8节点, 每节点3坐标
    K: DMatrix<f64>,                       // 单元刚度矩阵
    F: DVector<f64>,                       // 右端向量
}

impl Cube<8> {
    pub fn new(node_dof: usize) -> Self {
        Cube {
            node_dof,
            connectivity: [0; 8],
            nodes_coordinates: SMatrix::zeros(),
            K: DMatrix::zeros(8 * node_dof, 8 * node_dof),
            F: DVector::zeros(8 * node_dof),
        }
    }
}

impl<const N: usize> ElementBase<N, 3> for Cube<N> {
    fn node_dof(&self) -> usize {
        self.node_dof
    }

    fn node_count(&self) -> usize {
        self.connectivity.len()
    }

    fn connectivity(&self) -> &[usize; N] {
        &self.connectivity
    }

    // fn connectivity_mut(&mut self) -> &mut [usize; N] {
    //     &mut self.connectivity
    // }

    fn nodes_coordinates(&self) -> &SMatrix<f64, N, 3> {
        &self.nodes_coordinates
    }

    // fn nodes_coordinates_mut(&mut self) -> &mut SMatrix<f64, N, 3> {
    //     &mut self.nodes_coordinates
    // }

    fn K_mut(&mut self) -> &mut DMatrix<f64> {
        &mut self.K
    }

    fn F_mut(&mut self) -> &mut DVector<f64> {
        &mut self.F
    }
}

impl<const N: usize> GeneralElement<N, 3> for Cube<N> {
    fn update(
        &mut self,
        element_number: usize,                // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &MatrixXx3<f64>,   // 全局节点-坐标矩阵
    ) {
        connectivity_matrix
            .row(element_number)
            .iter()
            .enumerate()
            .for_each(|(idx, node_idx)| {
                self.connectivity[idx] = *node_idx;
            });

        self.connectivity
            .iter()
            .enumerate()
            .for_each(|(idx, node_idx)| {
                let row = coordinate_matrix.row(*node_idx);
                self.nodes_coordinates.set_row(idx, &row.fixed_columns(0));
            });
    }
}

#[cfg(test)]
mod tests {}
