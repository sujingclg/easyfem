use std::ops::AddAssign;

use nalgebra::{DMatrix, DVector, Matrix2x1, MatrixXx2, MatrixXx3, Vector2};

use crate::{
    base::{gauss::get_gauss_1d_matrix, utils::flatten_vector},
    materials::Material,
};

use super::{GaussResult, GeneralElement, StructureElement};

// TODO: 尝试用构建器模式将K和F矩阵变回静态矩阵
pub struct Edge2 {
    node_dof: usize,                   // 节点自由度
    connectivity: [usize; 2],          // 单元的节点序号数组
    nodes_coordinates: Matrix2x1<f64>, // 单元节点的全局坐标数组, 每单元2节点, 每节点1坐标
    gauss_matrix: MatrixXx2<f64>,      // 高斯积分矩阵, 1列->w 2列->xi
    K: DMatrix<f64>,                   // 单元刚度矩阵
    F: DVector<f64>,                   // 右端向量
}

impl Edge2 {
    pub fn new(gauss_deg: usize, node_dof: usize) -> Self {
        Edge2 {
            node_dof,
            connectivity: [0, 0],
            nodes_coordinates: Matrix2x1::zeros(),
            gauss_matrix: get_gauss_1d_matrix(gauss_deg),
            K: DMatrix::zeros(2 * node_dof, 2 * node_dof),
            F: DVector::zeros(2 * node_dof),
        }
    }

    fn gauss_point_calculate(&self, xi: f64) -> GaussResult<2, 1> {
        // 1维2节点等参元形函数
        let shp_val = Vector2::new(
            (1.0 - xi) / 2.0, // N1
            (1.0 + xi) / 2.0, // N2
        );

        // 梯度矩阵，每个元素分别是形函数对xi求偏导: \frac{\partial Ni}{\partial \xi}
        let mut shp_grad = Matrix2x1::new(
            -0.5, // dN1/dxi
            0.5,  // dN2/dxi
        );

        let mut dx_dxi = 0.0;

        // 遍历单元的每个节点
        for i in 0..self.connectivity.len() {
            dx_dxi += shp_grad[i] * self.nodes_coordinates[i];
        }
        // let det_jacob = f64::abs(dx_dxi);
        let det_J = dx_dxi; // jacob行列式

        shp_grad /= det_J; // 梯度矩阵除以jacob行列式以便进行单元组装

        GaussResult {
            shp_val,
            shp_grad,
            det_J,
        }
    }
}

impl GeneralElement<2, 1> for Edge2 {
    fn update(
        &mut self,
        element_number: usize,                   // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>,    // 全局单元-节点编号矩阵, 每单元2节点
        node_coordinate_matrix: &MatrixXx3<f64>, // 全局节点-坐标矩阵, 每节点3坐标只取第一个
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
                let row = node_coordinate_matrix.row(*node_idx);
                self.nodes_coordinates.set_row(idx, &row.fixed_columns(0)); // 每节点3坐标只取第一个
            });
    }

    fn general_stiffness_calculate<K, F>(&mut self, kij_operator: K, fi_operator: F)
    where
        K: Fn(usize, usize, &GaussResult<2, 1>) -> DMatrix<f64>,
        F: Fn(usize, &GaussResult<2, 1>) -> DVector<f64>,
    {
        for row in self.gauss_matrix.row_iter() {
            let xi = row[1];
            let w = row[0];
            let gauss_result = self.gauss_point_calculate(xi);
            let JxW = gauss_result.det_J * w;
            for i in 0..self.connectivity.len() {
                let fi = fi_operator(i, &gauss_result) * JxW;
                if fi.shape() != (self.node_dof, 1) {
                    // TODO: 尝试用范型处理
                    panic!("Shape of fi not match")
                }
                self.F
                    .rows_mut(i * self.node_dof, self.node_dof)
                    .add_assign(&fi);
                for j in 0..self.connectivity.len() {
                    let kij = kij_operator(i, j, &gauss_result) * JxW;
                    if kij.shape() != (self.node_dof, self.node_dof) {
                        // TODO: 尝试用范型处理
                        panic!("Shape of kij not match")
                    }
                    self.K
                        .view_mut((i, j), (self.node_dof, self.node_dof))
                        .add_assign(&kij);
                }
            }
        }
    }

    fn general_assemble(
        &mut self,
        stiffness_matrix: &mut DMatrix<f64>,
        right_vector: &mut DVector<f64>,
    ) {
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

impl StructureElement<1> for Edge2 {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<1>) {
        for row in self.gauss_matrix.row_iter() {
            let xi = row[1];
            let w = row[0];
            let gauss_result = self.gauss_point_calculate(xi);
            let JxW = gauss_result.det_J * w;
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

    fn structure_assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>) {
        for (i, node_i) in self.connectivity.iter().enumerate() {
            for (j, node_j) in self.connectivity.iter().enumerate() {
                stiffness_matrix[(*node_i, *node_j)] += self.K[(i, j)];
            }
        }
        self.K.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange1DMesh, Mesh};
    use nalgebra::SMatrix;

    use crate::materials::IsotropicLinearElastic1D;

    use super::*;

    #[test]
    fn structure_test_1() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 5, "edge2");
        let mut edge2 = Edge2::new(2, 1);
        let mut stiffness_matrix = DMatrix::zeros(6, 6);
        let mat = IsotropicLinearElastic1D::new(1.0e9, 1.0);
        for element_number in 0..mesh.element_count() {
            edge2.update(element_number, mesh.elements(), mesh.nodes());
            edge2.structure_stiffness_calculate(&mat);
            edge2.structure_assemble(&mut stiffness_matrix);
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
    fn poisson_test_1() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 10, "edge2");
        let mut edge2 = Edge2::new(2, 1);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);

        fn kij_operator(i: usize, j: usize, gauss_result: &GaussResult<2, 1>) -> DMatrix<f64> {
            let GaussResult { shp_grad, .. } = gauss_result;
            DMatrix::from_element(1, 1, shp_grad[j] * shp_grad[i])
        }

        fn fi_operator(i: usize, gauss_result: &GaussResult<2, 1>) -> DVector<f64> {
            let GaussResult { shp_val, .. } = gauss_result;
            DVector::from_element(1, 200.0 * shp_val[i])
        }

        for element_number in 0..mesh.element_count() {
            edge2.update(element_number, mesh.elements(), mesh.nodes());
            edge2.general_stiffness_calculate(kij_operator, fi_operator);
            edge2.general_assemble(&mut stiffness_matrix, &mut right_vector);
        }

        // boundary conditions
        stiffness_matrix[(0, 0)] += 1e16;
        right_vector[0] += 0.5 * 1e16;
        stiffness_matrix[(n_dofs - 1, n_dofs - 1)] += 1e16;
        right_vector[n_dofs - 1] += 5.0 * 1e16;

        let answer = SMatrix::<f64, 11, 1>::from_column_slice(&[
            0.50, 9.95, 17.40, 22.85, 26.30, 27.75, 27.20, 24.65, 20.10, 13.55, 5.00,
        ]);
        let err = 1e-13;
        if let Some(phi) = stiffness_matrix.lu().solve(&right_vector) {
            println!("{:.2}", &phi);
            assert!(phi.relative_eq(&answer, err, err));
        }
    }
}
