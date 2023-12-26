use nalgebra::{DMatrix, DVector, MatrixXx3, SMatrix};

use super::{ElementBase, GeneralElement};

pub type Edge2 = Edge<2>;
pub type Edge3 = Edge<3>;
pub type Edge4 = Edge<4>;

/// N -> 单元节点个数
pub struct Edge<const N: usize> {
    node_dof: usize,                       // 节点自由度
    connectivity: [usize; N],              // 单元的节点序号数组
    nodes_coordinates: SMatrix<f64, N, 1>, // 单元节点的全局坐标数组, 每单元2节点, 每节点1坐标
    K: DMatrix<f64>,                       // 单元刚度矩阵
    F: DVector<f64>,                       // 右端向量
}

impl Edge2 {
    pub fn new(node_dof: usize) -> Self {
        Edge {
            node_dof,
            connectivity: [0, 0],
            nodes_coordinates: SMatrix::zeros(),
            K: DMatrix::zeros(2 * node_dof, 2 * node_dof),
            F: DVector::zeros(2 * node_dof),
        }
    }
}

impl Edge3 {
    pub fn new(node_dof: usize) -> Self {
        Edge {
            node_dof,
            connectivity: [0; 3],
            nodes_coordinates: SMatrix::zeros(),
            K: DMatrix::zeros(3 * node_dof, 3 * node_dof),
            F: DVector::zeros(3 * node_dof),
        }
    }
}

impl Edge4 {
    pub fn new(node_dof: usize) -> Self {
        Edge {
            node_dof,
            connectivity: [0; 4],
            nodes_coordinates: SMatrix::zeros(),
            K: DMatrix::zeros(4 * node_dof, 4 * node_dof),
            F: DVector::zeros(4 * node_dof),
        }
    }
}

impl<const N: usize> ElementBase<N> for Edge<N> {
    type CoordMatrix = SMatrix<f64, N, 1>;

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

    fn nodes_coordinates(&self) -> &Self::CoordMatrix {
        &self.nodes_coordinates
    }
    // fn nodes_coordinates_mut(&mut self) -> &mut SMatrix<f64, N, 1> {
    //     &mut self.nodes_coordinates
    // }

    fn K_mut(&mut self) -> &mut DMatrix<f64> {
        &mut self.K
    }

    fn F_mut(&mut self) -> &mut DVector<f64> {
        &mut self.F
    }
}

impl<const N: usize> GeneralElement<N> for Edge<N> {
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
                self.nodes_coordinates.set_row(idx, &row.fixed_columns(0)); // 每节点3坐标只取第一个
            });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn edge2_test() {
        // TODO:
    }
}
