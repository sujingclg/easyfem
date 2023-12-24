use nalgebra::{DMatrix, DVector, MatrixXx3, SMatrix};

use crate::{base::utils::flatten_vector, gauss::GaussEdge, materials::Material};

use super::{GeneralElement, StructureElement};

pub type Edge2 = Edge<2>;
pub type Edge3 = Edge<3>;

/// N -> 单元节点个数
pub struct Edge<const N: usize> {
    node_dof: usize,                       // 节点自由度
    connectivity: [usize; N],              // 单元的节点序号数组
    nodes_coordinates: SMatrix<f64, N, 1>, // 单元节点的全局坐标数组, 每单元2节点, 每节点1坐标
    gauss: GaussEdge,                      // 高斯积分算子
    K: DMatrix<f64>,                       // 单元刚度矩阵
    F: DVector<f64>,                       // 右端向量
}

impl Edge<2> {
    pub fn new(node_dof: usize, gauss: GaussEdge) -> Self {
        Edge {
            node_dof,
            connectivity: [0, 0],
            nodes_coordinates: SMatrix::zeros(),
            gauss,
            K: DMatrix::zeros(2 * node_dof, 2 * node_dof),
            F: DVector::zeros(2 * node_dof),
        }
    }
}

impl Edge<3> {
    pub fn new(node_dof: usize, gauss: GaussEdge) -> Self {
        Edge {
            node_dof,
            connectivity: [0; 3],
            nodes_coordinates: SMatrix::zeros(),
            gauss,
            K: DMatrix::zeros(3 * node_dof, 3 * node_dof),
            F: DVector::zeros(3 * node_dof),
        }
    }
}

impl<const N: usize> GeneralElement for Edge<N> {
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

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>, right_vector: &mut DVector<f64>) {
        let flattened_connectivity = flatten_vector(&self.connectivity, self.node_dof);
        for (i, node_i) in flattened_connectivity.iter().enumerate() {
            right_vector[*node_i] += self.F[i];
            for (j, node_j) in flattened_connectivity.iter().enumerate() {
                stiffness_matrix[(*node_i, *node_j)] += self.K[(i, j)];
            }
        }
        self.K.fill(0.0);
        self.F.fill(0.0);
    }
}

impl StructureElement<1> for Edge<2> {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<1>) {
        for row in self.gauss.get_gauss_matrix().row_iter() {
            let xi = row[1];
            let w = row[0];
            let gauss_result = self
                .gauss
                .linear_shape_func_calc(&self.nodes_coordinates, [xi]);
            let JxW = gauss_result.det_j * w;
            for i in 0..self.connectivity.len() {
                for j in 0..self.connectivity.len() {
                    // 这里要对高斯积分进行累加
                    self.K[(i, j)] += gauss_result.shp_grad[j]
                        * gauss_result.shp_grad[i]
                        * mat.get_constitutive_matrix()[(0, 0)]
                        * JxW;
                }
            }
        }
    }
}

impl StructureElement<1> for Edge<3> {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<1>) {
        for row in self.gauss.get_gauss_matrix().row_iter() {
            let xi = row[1];
            let w = row[0];
            let gauss_result = self
                .gauss
                .square_shape_func_calc(&self.nodes_coordinates, [xi]);
            let JxW = gauss_result.det_j * w;
            for i in 0..self.connectivity.len() {
                for j in 0..self.connectivity.len() {
                    // 这里要对高斯积分进行累加
                    self.K[(i, j)] += gauss_result.shp_grad[j]
                        * gauss_result.shp_grad[i]
                        * mat.get_constitutive_matrix()[(0, 0)]
                        * JxW;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange1DMesh, Mesh};
    use nalgebra::SMatrix;

    use crate::materials::IsotropicLinearElastic1D;

    use super::*;

    #[test]
    fn structure_edge2_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 5, "edge2");
        let mut edge2 = Edge2::new(1, GaussEdge::new(2));
        let n_dofs = mesh.get_node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic1D::new(1.0e9, 1.0);
        for element_number in 0..mesh.get_element_count() {
            edge2.update(element_number, mesh.get_elements(), mesh.get_nodes());
            edge2.structure_stiffness_calculate(&mat);
            edge2.assemble(&mut stiffness_matrix, &mut right_vector);
        }
        println!("{:.3e}", stiffness_matrix);
        let answer = SMatrix::<f64, 6, 6>::new(
            5.000e9, -5.000e9, 0.000e0, 0.000e0, 0.000e0, 0.000e0, -5.000e9, 1.000e10, -5.000e9,
            0.000e0, 0.000e0, 0.000e0, 0.000e0, -5.000e9, 1.000e10, -5.000e9, 0.000e0, 0.000e0,
            0.000e0, 0.000e0, -5.000e9, 1.000e10, -5.000e9, 0.000e0, 0.000e0, 0.000e0, 0.000e0,
            -5.000e9, 1.000e10, -5.000e9, 0.000e0, 0.000e0, 0.000e0, 0.000e0, -5.000e9, 5.000e9,
        );
        let err = 1e-15;
        assert!(stiffness_matrix.relative_eq(&answer, err, err));
    }

    #[test]
    fn structure_edge3_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 2, "edge3");
        let mut edge3 = Edge3::new(1, GaussEdge::new(2));
        let n_dofs = mesh.get_node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic1D::new(1.0, 1.0);

        for element_number in 0..mesh.get_element_count() {
            edge3.update(element_number, mesh.get_elements(), mesh.get_nodes());
            edge3.structure_stiffness_calculate(&mat);
            edge3.assemble(&mut stiffness_matrix, &mut right_vector);
        }
        println!("{:.3}", stiffness_matrix);
        let answer = SMatrix::<f64, 5, 5>::new(
            4.667, -5.333, 0.667, 0.000, 0.000, -5.333, 10.667, -5.333, 0.000, 0.000, 0.667,
            -5.333, 9.333, -5.333, 0.667, 0.000, 0.000, -5.333, 10.667, -5.333, 0.000, 0.000,
            0.667, -5.333, 4.667,
        );
        let err = 1e-3;
        assert!(stiffness_matrix.relative_eq(&answer, err, err));
    }

    #[test]
    fn poisson_edge2_test() {
        // TODO:
    }
}
