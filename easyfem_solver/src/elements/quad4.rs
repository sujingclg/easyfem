use std::ops::AddAssign;

use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix3x2, Matrix4x2, MatrixXx3, Vector4};

use crate::{
    base::{gauss::get_gauss_2d_matrix, utils::flatten_vector},
    materials::Material,
};

use super::{GaussResult, GeneralElement, StructureElement};

pub struct Quad4 {
    node_dof: usize,                   // 节点自由度 结构分析为2
    connectivity: [usize; 4],          // 单元的节点序号数组
    nodes_coordinates: Matrix4x2<f64>, // 单元节点的全局坐标数组, 每单元4节点, 每节点2坐标
    gauss_matrix: MatrixXx3<f64>,      // 高斯积分矩阵, 1列->w 2列->xi 3列->eta
    K: DMatrix<f64>,                   // 单元刚度矩阵
    F: DVector<f64>,                   // 右端向量
}

impl Quad4 {
    pub fn new(gauss_deg: usize, node_dof: usize) -> Self {
        Quad4 {
            node_dof,
            connectivity: [0; 4],
            nodes_coordinates: Matrix4x2::zeros(),
            gauss_matrix: get_gauss_2d_matrix(gauss_deg),
            K: DMatrix::zeros(4 * node_dof, 4 * node_dof),
            F: DVector::zeros(4 * node_dof),
        }
    }

    // 在每个高斯点上做个预计算
    fn gauss_point_calculate(&self, xi: f64, eta: f64) -> GaussResult<4, 2> {
        // 2维4节点等参元形函数
        let shp_val = Vector4::new(
            (1.0 - xi) * (1.0 - eta) / 4.0, // N1
            (1.0 + xi) * (1.0 - eta) / 4.0, // N2
            (1.0 + xi) * (1.0 + eta) / 4.0, // N3
            (1.0 - xi) * (1.0 + eta) / 4.0, // N4
        );

        // 梯度矩阵，每个元素分别是形函数分别对xi和eta求偏导: \frac{\partial Ni}{\partial \xi} \frac{\partial Ni}{\partial \eta}
        let mut shp_grad = Matrix4x2::new(
            (eta - 1.0) / 4.0,  // dN1/dxi
            (xi - 1.0) / 4.0,   // dN1/deta
            (1.0 - eta) / 4.0,  // dN2/dxi
            -(1.0 + xi) / 4.0,  // dN2/deta
            (1.0 + eta) / 4.0,  // dN3/dxi
            (1.0 + xi) / 4.0,   // dN3/deta
            -(1.0 + eta) / 4.0, // dN4/xi
            (1.0 - xi) / 4.0,   // dN4/deta
        );

        // jacob矩阵形式为[[dx_dxi, dy_dxi],[dx_deta, dy_deta]]
        let J = shp_grad.transpose() * self.nodes_coordinates;
        let det_J = J.determinant(); // 行列式

        // jacob逆矩阵形式为[[dxi_dx, deta_dx],[dxi_dy, deta_dy]]
        let inverse_J = J.try_inverse().unwrap(); // 逆矩阵

        // shp_grad 的每一行变为下式, Ni从N1到N4
        // dNi/dxi * dxi/dx + dNi/deta * deta/dx, dNi/dxi * dxi/dy + dNi/deta * deta/dy
        shp_grad = shp_grad * inverse_J.transpose();

        GaussResult {
            shp_val,
            shp_grad,
            det_J,
        }
    }
}

impl GeneralElement<4, 2> for Quad4 {
    fn update(
        &mut self,
        element_number: usize,                   // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>,    // 全局单元-节点编号矩阵, 每单元4节点
        node_coordinate_matrix: &MatrixXx3<f64>, // 全局节点-坐标矩阵, 每节点3坐标只取前两个
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
                self.nodes_coordinates.set_row(idx, &row.fixed_columns(0)) // 每节点3坐标只取前两个
            });
    }

    fn general_stiffness_calculate<K, F>(&mut self, kij_operator: K, fi_operator: F)
    where
        K: Fn(usize, usize, &GaussResult<4, 2>) -> DMatrix<f64>,
        F: Fn(usize, &GaussResult<4, 2>) -> DVector<f64>,
    {
        for row in self.gauss_matrix.row_iter() {
            let xi = row[1];
            let eta = row[2];
            let w = row[0];
            let gauss_result = self.gauss_point_calculate(xi, eta);
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

impl StructureElement<3> for Quad4 {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<3>) {
        let mut B = Matrix3x2::zeros(); // 应变矩阵
        let mut Bt = Matrix2x3::zeros(); // 应变矩阵的转置
        for row in self.gauss_matrix.row_iter() {
            let xi = row[1];
            let eta = row[2];
            let w = row[0];
            let GaussResult {
                shp_grad, det_J, ..
            } = self.gauss_point_calculate(xi, eta);
            let JxW = det_J * w;
            for i in 0..self.connectivity.len() {
                B[(0, 0)] = shp_grad[(i, 0)]; // 矩阵分块乘法, 每次计算出2x2的矩阵, 然后组装到单元刚度矩阵的对应位置
                B[(1, 1)] = shp_grad[(i, 1)];
                B[(2, 0)] = shp_grad[(i, 1)];
                B[(2, 1)] = shp_grad[(i, 0)];
                for j in 0..self.connectivity.len() {
                    Bt[(0, 0)] = shp_grad[(j, 0)];
                    Bt[(0, 2)] = shp_grad[(j, 1)];
                    Bt[(1, 1)] = shp_grad[(j, 1)];
                    Bt[(1, 2)] = shp_grad[(j, 0)];
                    let C = Bt * mat.get_constitutive_matrix() * B;
                    // 这里要对高斯积分进行累加
                    self.K[(2 * i + 0, 2 * j + 0)] += C[(0, 0)] * JxW; // K_ux,ux
                    self.K[(2 * i + 0, 2 * j + 1)] += C[(0, 1)] * JxW; // K_ux,uy
                    self.K[(2 * i + 1, 2 * j + 0)] += C[(1, 0)] * JxW; // K_uy,ux
                    self.K[(2 * i + 1, 2 * j + 1)] += C[(1, 1)] * JxW; // K_uy,uy
                }
            }
        }
    }

    fn structure_assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>) {
        for (i, node_i) in self.connectivity.iter().enumerate() {
            for (j, node_j) in self.connectivity.iter().enumerate() {
                stiffness_matrix[(2 * node_i + 0, 2 * node_j + 0)] +=
                    self.K[(2 * i + 0, 2 * j + 0)];
                stiffness_matrix[(2 * node_i + 0, 2 * node_j + 1)] +=
                    self.K[(2 * i + 0, 2 * j + 1)];
                stiffness_matrix[(2 * node_i + 1, 2 * node_j + 0)] +=
                    self.K[(2 * i + 1, 2 * j + 0)];
                stiffness_matrix[(2 * node_i + 1, 2 * node_j + 1)] +=
                    self.K[(2 * i + 1, 2 * j + 1)];
            }
        }
        self.K.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange2DMesh, Mesh};
    use nalgebra::SMatrix;

    use crate::materials::{IsotropicLinearElastic2D, PlaneCondition::*};

    use super::*;

    #[test]
    /// Daryl L. Logan 有限元方法基础教程(第五版)  例10.4
    fn structure_test_1() {
        let n_dofs: usize = 8;
        let mesh = Lagrange2DMesh::new(3.0, 5.0, 1, 2.0, 4.0, 1, "quad4");
        println!("{}", mesh);
        let mut quad4 = Quad4::new(2, 2);
        let mat = IsotropicLinearElastic2D::new(3.0e7, 0.25, PlaneStress, 1.0);
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        for element_number in 0..mesh.get_element_count() {
            quad4.update(element_number, mesh.get_elements(), mesh.get_nodes());
            quad4.structure_stiffness_calculate(&mat);
            quad4.structure_assemble(&mut stiffness_matrix);
        }
        println!("stiffness_matrix = {:.3e}", stiffness_matrix);
        let answer = SMatrix::<f64, 8, 8>::from_row_slice(&[
            1.467e7, 5.000e6, -8.667e6, 1.000e6, 1.333e6, -1.000e6, -7.333e6, -5.000e6, 5.000e6,
            1.467e7, -1.000e6, 1.333e6, 1.000e6, -8.667e6, -5.000e6, -7.333e6, -8.667e6, -1.000e6,
            1.467e7, -5.000e6, -7.333e6, 5.000e6, 1.333e6, 1.000e6, 1.000e6, 1.333e6, -5.000e6,
            1.467e7, 5.000e6, -7.333e6, -1.000e6, -8.667e6, 1.333e6, 1.000e6, -7.333e6, 5.000e6,
            1.467e7, -5.000e6, -8.667e6, -1.000e6, -1.000e6, -8.667e6, 5.000e6, -7.333e6, -5.000e6,
            1.467e7, 1.000e6, 1.333e6, -7.333e6, -5.000e6, 1.333e6, -1.000e6, -8.667e6, 1.000e6,
            1.467e7, 5.000e6, -5.000e6, -7.333e6, 1.000e6, -8.667e6, -1.000e6, 1.333e6, 5.000e6,
            1.467e7,
        ]);
        let err = 1e-3;
        assert!(stiffness_matrix.relative_eq(&answer, err, err));
    }

    #[test]
    fn poisson_test_1() {
        let mesh = Lagrange2DMesh::new(0.0, 1.0, 5, 0.0, 1.0, 5, "quad4");
        let mut quad4 = Quad4::new(2, 1);
        let n_dofs = mesh.get_node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        for element_number in 0..mesh.get_element_count() {
            quad4.update(element_number, mesh.get_elements(), mesh.get_nodes());
            quad4.general_stiffness_calculate(
                |i, j, gauss_result| {
                    let GaussResult { shp_grad, .. } = gauss_result;
                    DMatrix::from_element(
                        1,
                        1,
                        shp_grad[(j, 0)] * shp_grad[(i, 0)] + shp_grad[(j, 1)] * shp_grad[(i, 1)],
                    )
                },
                |i, gauss_result| {
                    let GaussResult { shp_val, .. } = gauss_result;
                    DVector::from_element(1, 200.0 * shp_val[i])
                },
            );
            quad4.general_assemble(&mut stiffness_matrix, &mut right_vector);
        }

        // boundary conditions
        if let Some(leftnodes) = mesh.get_boundary_node_ids().get("left") {
            for id in leftnodes {
                stiffness_matrix[(*id, *id)] += 1e16;
                right_vector[*id] += 5.0 * 1e16;
            }
        }
        if let Some(rightnodes) = mesh.get_boundary_node_ids().get("right") {
            for id in rightnodes {
                stiffness_matrix[(*id, *id)] += 1e16;
                right_vector[*id] += 5.0 * 1e16;
            }
        }
        if let Some(rightnodes) = mesh.get_boundary_node_ids().get("top") {
            for id in rightnodes {
                stiffness_matrix[(*id, *id)] += 1e16;
                right_vector[*id] += 5.0 * 1e16;
            }
        }
        if let Some(rightnodes) = mesh.get_boundary_node_ids().get("bottom") {
            for id in rightnodes {
                stiffness_matrix[(*id, *id)] += 1e16;
                right_vector[*id] += 5.0 * 1e16;
            }
        }

        let answer = SMatrix::<f64, 36, 1>::from_column_slice(&[
            5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 12.263, 14.947, 14.947, 12.263, 5.000,
            5.000, 14.947, 19.211, 19.211, 14.947, 5.000, 5.000, 14.947, 19.211, 19.211, 14.947,
            5.000, 5.000, 12.263, 14.947, 14.947, 12.263, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000,
            5.000,
        ]);
        let err = 1e-4;
        if let Some(phi) = stiffness_matrix.lu().solve(&right_vector) {
            println!(
                "{:.3}",
                SMatrix::<f64, 6, 6>::from_column_slice(&phi.as_slice())
            );
            assert!(phi.relative_eq(&answer, err, err));
        }
    }
}
