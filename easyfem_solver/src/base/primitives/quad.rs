use nalgebra::{DMatrix, DVector, Matrix4x2, MatrixXx3, SMatrix};

use super::{GeneralElement, PrimitiveBase};

// pub type Quad4 = Quad<4>;
// pub type Quad9 = Quad<9>;

/// N -> 单元节点个数
pub struct Quad<const N: usize> {
    node_dof: usize,                       // 节点自由度
    connectivity: [usize; N],              // 单元的节点序号数组
    nodes_coordinates: SMatrix<f64, N, 2>, // 单元节点的全局坐标数组, 每单元2节点, 每节点1坐标
    K: DMatrix<f64>,                       // 单元刚度矩阵
    F: DVector<f64>,                       // 右端向量
}

impl Quad<4> {
    pub fn new(node_dof: usize) -> Self {
        Quad {
            node_dof,
            connectivity: [0; 4],
            nodes_coordinates: Matrix4x2::zeros(),
            // gauss,
            K: DMatrix::zeros(4 * node_dof, 4 * node_dof),
            F: DVector::zeros(4 * node_dof),
        }
    }
}

impl Quad<9> {
    pub fn new(node_dof: usize) -> Self {
        Quad {
            node_dof,
            connectivity: [0; 9],
            nodes_coordinates: SMatrix::zeros(),
            // gauss,
            K: DMatrix::zeros(9 * node_dof, 9 * node_dof),
            F: DVector::zeros(9 * node_dof),
        }
    }
}

impl<const N: usize> PrimitiveBase<N, 2> for Quad<N> {
    fn node_dof(&self) -> usize {
        self.node_dof
    }

    fn node_count(&self) -> usize {
        self.connectivity.len()
    }

    fn connectivity(&self) -> &[usize; N] {
        &self.connectivity
    }

    fn nodes_coordinates(&self) -> &SMatrix<f64, N, 2> {
        &self.nodes_coordinates
    }

    fn K_mut(&mut self) -> &mut DMatrix<f64> {
        &mut self.K
    }

    fn F_mut(&mut self) -> &mut DVector<f64> {
        &mut self.F
    }
}

impl<const N: usize> GeneralElement<N, 2> for Quad<N> {
    fn update(
        &mut self,
        element_number: usize,                // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>, // 全局单元-节点编号矩阵, 每单元4节点
        coordinate_matrix: &MatrixXx3<f64>,   // 全局节点-坐标矩阵, 每节点3坐标只取前两个
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
                self.nodes_coordinates.set_row(idx, &row.fixed_columns(0)) // 每节点3坐标只取前两个
            });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn quad4_test() {
        // TODO:
    }
}
