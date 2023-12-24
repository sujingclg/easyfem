use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix3x2, Matrix4x2, MatrixXx3, SMatrix};

use crate::{
    base::utils::flatten_vector,
    gauss::{GaussQuad, GaussResult},
    materials::Material,
};

use super::{GeneralElement, StructureElement};

pub type Quad4 = Quad<4>;
pub type Quad9 = Quad<9>;

/// N -> 单元节点个数
pub struct Quad<const N: usize> {
    node_dof: usize,                       // 节点自由度
    connectivity: [usize; N],              // 单元的节点序号数组
    nodes_coordinates: SMatrix<f64, N, 2>, // 单元节点的全局坐标数组, 每单元2节点, 每节点1坐标
    gauss: GaussQuad,                      // 高斯积分算子
    K: DMatrix<f64>,                       // 单元刚度矩阵
    F: DVector<f64>,                       // 右端向量
}

impl Quad<4> {
    pub fn new(node_dof: usize, gauss: GaussQuad) -> Self {
        Quad {
            node_dof,
            connectivity: [0; 4],
            nodes_coordinates: Matrix4x2::zeros(),
            gauss,
            K: DMatrix::zeros(4 * node_dof, 4 * node_dof),
            F: DVector::zeros(4 * node_dof),
        }
    }
}

impl Quad<9> {
    pub fn new(node_dof: usize, gauss: GaussQuad) -> Self {
        Quad {
            node_dof,
            connectivity: [0; 9],
            nodes_coordinates: SMatrix::zeros(),
            gauss,
            K: DMatrix::zeros(9 * node_dof, 9 * node_dof),
            F: DVector::zeros(9 * node_dof),
        }
    }
}

impl<const N: usize> GeneralElement for Quad<N> {
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

impl StructureElement<3> for Quad<4> {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<3>) {
        let mut B = Matrix3x2::zeros(); // 应变矩阵
        let mut Bt = Matrix2x3::zeros(); // 应变矩阵的转置
        for row in self.gauss.get_gauss_matrix().row_iter() {
            let xi = row[1];
            let eta = row[2];
            let w = row[0];
            let GaussResult {
                shp_grad, det_j, ..
            } = self
                .gauss
                .linear_shape_func_calc(&self.nodes_coordinates, [xi, eta]);
            let JxW = det_j * w;
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
}

impl StructureElement<3> for Quad<9> {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<3>) {
        let mut B = Matrix3x2::zeros(); // 应变矩阵
        let mut Bt = Matrix2x3::zeros(); // 应变矩阵的转置
        for row in self.gauss.get_gauss_matrix().row_iter() {
            let xi = row[1];
            let eta = row[2];
            let w = row[0];
            let GaussResult {
                shp_grad, det_j, ..
            } = self
                .gauss
                .square_shape_func_calc(&self.nodes_coordinates, [xi, eta]);
            let JxW = det_j * w;
            for i in 0..self.connectivity.len() {
                B[(0, 0)] = shp_grad[(i, 0)]; // 矩阵分块乘法, 每次计算出2x2的矩阵, 然后组装到单元刚度矩阵的对应位置
                B[(1, 1)] = shp_grad[(i, 1)];
                B[(2, 0)] = shp_grad[(i, 1)];
                B[(2, 1)] = shp_grad[(i, 0)];
                for j in 0..self.connectivity.len() {
                    // println!("i = {i}, j = {j}");
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
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange2DMesh, Mesh};

    use crate::materials::{IsotropicLinearElastic2D, PlaneCondition::*};

    use super::*;

    #[test]
    /// Daryl L. Logan 有限元方法基础教程(第五版)  例10.4
    fn structure_quad4_test() {
        let mesh = Lagrange2DMesh::new(3.0, 5.0, 1, 2.0, 4.0, 1, "quad4");
        println!("{}", mesh);
        let n_dofs = mesh.get_node_count() * 2;
        let mut quad4 = Quad4::new(2, GaussQuad::new(2));
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic2D::new(3.0e7, 0.25, PlaneStress, 1.0);
        for element_number in 0..mesh.get_element_count() {
            quad4.update(element_number, mesh.get_elements(), mesh.get_nodes());
            quad4.structure_stiffness_calculate(&mat);
            quad4.assemble(&mut stiffness_matrix, &mut right_vector);
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
    fn structure_quad9_test() {
        let n_dofs: usize = 18;
        let connectivity_matrix = DMatrix::from_row_slice(1, 9, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let node_coordinate_matrix = MatrixXx3::from_row_slice(&[
            3.0, 2.0, 0.0, // 0
            5.0, 2.0, 0.0, // 1
            5.0, 4.0, 0.0, // 2
            3.0, 4.0, 0.0, // 3
            4.0, 2.0, 0.0, // 4
            5.0, 3.0, 0.0, // 5
            4.0, 4.0, 0.0, // 6
            3.0, 3.0, 0.0, // 7
            4.0, 3.0, 0.0, // 8
        ]);
        let mut quad9 = Quad9::new(2, GaussQuad::new(2));
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic2D::new(3.0e7, 0.25, PlaneStress, 1.0);
        for element_number in 0..connectivity_matrix.nrows() {
            quad9.update(
                element_number,
                &connectivity_matrix,
                &node_coordinate_matrix,
            );
            quad9.structure_stiffness_calculate(&mat);
            quad9.assemble(&mut stiffness_matrix, &mut right_vector);
        }
        println!("stiffness_matrix = {:.3e}", stiffness_matrix);
    }

    #[test]
    fn poisson_quad4_test() {
        // TODO:
    }
}
