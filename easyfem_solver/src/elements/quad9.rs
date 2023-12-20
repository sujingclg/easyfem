use nalgebra::{DMatrix, Matrix2x3, Matrix3x2, MatrixXx3, SMatrix};

use crate::{base::gauss::get_gauss_2d_matrix, materials::Material};

use super::{GeneralElement, StructureElement};

pub struct Quad9 {
    nodes_numbers: [usize; 9],             // 单元的节点序号数组
    nodes_coordinates: SMatrix<f64, 9, 2>, // 单元节点的全局坐标数组, 每单元9节点, 每节点2坐标
    gauss_matrix: MatrixXx3<f64>,          // 高斯积分矩阵, 1列->w 2列->xi 3列->eta
    K: SMatrix<f64, 18, 18>,               // 单元刚度矩阵
}

impl Quad9 {
    pub fn new(gauss_deg: usize) -> Self {
        Quad9 {
            nodes_numbers: [0; 9],
            nodes_coordinates: SMatrix::zeros(),
            gauss_matrix: get_gauss_2d_matrix(gauss_deg),
            K: SMatrix::zeros(),
        }
    }

    fn gauss_point_calculate(
        &self,
        xi: f64,
        eta: f64,
    ) -> (SMatrix<f64, 9, 1>, SMatrix<f64, 9, 2>, f64) {
        // 2维9节点等参元形函数
        let shape_function_matrix = SMatrix::from_column_slice(&[
            (xi * xi - xi) * (eta * eta - eta) / 4.0,  // N1
            (xi * xi + xi) * (eta * eta - eta) / 4.0,  // N2
            (xi * xi + xi) * (eta * eta + eta) / 4.0,  // N3
            (xi * xi - xi) * (eta * eta + eta) / 4.0,  // N4
            (1.0 - xi * xi) * (eta * eta - eta) / 2.0, // N5
            (xi * xi + xi) * (1.0 - eta * eta) / 2.0,  // N6
            (1.0 - xi * xi) * (eta * eta + eta) / 2.0, // N7
            (xi * xi - xi) * (1.0 - eta * eta) / 2.0,  // N8
            (1.0 - xi * xi) * (1.0 - eta * eta),       // N9
        ]);

        let mut gradient_matrix = SMatrix::<f64, 9, 2>::from_row_slice(&[
            (2.0 * xi - 1.0) * (eta * eta - eta) / 4.0, // dN1/dxi
            (xi * xi - xi) * (2.0 * eta - 1.0) / 4.0,   // dN1/deta
            //
            (2.0 * xi + 1.0) * (eta * eta - eta) / 4.0, // dN2/dxi
            (xi * xi + xi) * (2.0 * eta - 1.0) / 4.0,   // dN2/deta
            //
            (2.0 * xi + 1.0) * (eta * eta + eta) / 4.0, // dN3/dxi
            (xi * xi + xi) * (2.0 * eta + 1.0) / 4.0,   // dN3/deta
            //
            (2.0 * xi - 1.0) * (eta * eta + eta) / 4.0, // dN4/dxi
            (xi * xi - xi) * (2.0 * eta + 1.0) / 4.0,   // dN4/deta
            //
            -xi * (eta * eta - eta),                   // dN5/dxi
            (1.0 - xi * xi) * (2.0 * eta - 1.0) / 2.0, // dN5/deta
            //
            (2.0 * xi + 1.0) * (1.0 - eta * eta) / 2.0, // dN6/dxi
            -(xi * xi + xi) * eta,                      // dN6/deta
            //
            -xi * (eta * eta + eta),                   // dN7/dxi
            (1.0 - xi * xi) * (2.0 * eta + 1.0) / 2.0, // dN7/deta
            //
            (2.0 * xi - 1.0) * (1.0 - eta * eta) / 2.0, // dN8/dxi
            -(xi * xi - xi) * eta,                      // dN8/deta
            //
            -2.0 * xi * (1.0 - eta * eta), // dN9/dxi
            -2.0 * eta * (1.0 - xi * xi),  // dN9/deta
        ]);

        let J = gradient_matrix.transpose() * self.nodes_coordinates;
        let det_J = J.determinant();

        let inverse_J = J.try_inverse().unwrap();

        gradient_matrix = gradient_matrix * inverse_J.transpose();

        (shape_function_matrix, gradient_matrix, det_J)
    }
}

impl GeneralElement for Quad9 {
    // TODO: 代码重复，考虑合并
    fn update(
        &mut self,
        element_number: usize,                   // 单元编号, 即单元的全局索引
        element_node_matrix: &DMatrix<usize>,    // 全局单元-节点编号矩阵, 每单元4节点
        node_coordinate_matrix: &MatrixXx3<f64>, // 全局节点-坐标矩阵, 每节点3坐标只取前两个
    ) {
        element_node_matrix
            .row(element_number)
            .iter()
            .enumerate()
            .for_each(|(idx, node_idx)| {
                self.nodes_numbers[idx] = *node_idx;
            });

        self.nodes_numbers
            .iter()
            .enumerate()
            .for_each(|(idx, node_idx)| {
                let row = node_coordinate_matrix.row(*node_idx);
                self.nodes_coordinates.set_row(idx, &row.fixed_columns(0)) // 每节点3坐标只取前两个
            });
    }

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>) {
        // TODO: 代码重复，考虑合并
        for (i, node_i) in self.nodes_numbers.iter().enumerate() {
            for (j, node_j) in self.nodes_numbers.iter().enumerate() {
                // println!("i = {i}, node_i = {node_i}, j = {j}, node_j = {node_j}");
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

impl StructureElement<3> for Quad9 {
    fn structure_calculate(&mut self, mat: &impl Material<3>) {
        let mut B = Matrix3x2::zeros(); // 应变矩阵
        let mut Bt = Matrix2x3::zeros(); // 应变矩阵的转置
        for row in self.gauss_matrix.row_iter() {
            let xi = row[1];
            let eta = row[2];
            let w = row[0];
            let (_, gradient_matrix, det_J) = self.gauss_point_calculate(xi, eta);
            let JxW = det_J * w;
            for i in 0..self.nodes_numbers.len() {
                B[(0, 0)] = gradient_matrix[(i, 0)]; // 矩阵分块乘法, 每次计算出2x2的矩阵, 然后组装到单元刚度矩阵的对应位置
                B[(1, 1)] = gradient_matrix[(i, 1)];
                B[(2, 0)] = gradient_matrix[(i, 1)];
                B[(2, 1)] = gradient_matrix[(i, 0)];
                for j in 0..self.nodes_numbers.len() {
                    // println!("i = {i}, j = {j}");
                    Bt[(0, 0)] = gradient_matrix[(j, 0)];
                    Bt[(0, 2)] = gradient_matrix[(j, 1)];
                    Bt[(1, 1)] = gradient_matrix[(j, 1)];
                    Bt[(1, 2)] = gradient_matrix[(j, 0)];
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
    use nalgebra::{DMatrix, MatrixXx3};

    use crate::materials::{IsotropicLinearElastic2D, PlaneCondition};

    use super::*;

    #[test]
    fn structure_test_1() {
        let n_dofs: usize = 18;
        let element_node_matrix = DMatrix::from_row_slice(1, 9, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
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
        let mut quad4 = Quad9::new(2);
        let mat = IsotropicLinearElastic2D::new(3.0e7, 0.25, PlaneCondition::PlaneStress, 1.0);
        let mut stiffness_matrix = DMatrix::<f64>::zeros(n_dofs, n_dofs);
        for element_number in 0..element_node_matrix.nrows() {
            quad4.update(
                element_number,
                &element_node_matrix,
                &node_coordinate_matrix,
            );
            quad4.structure_calculate(&mat);
            quad4.assemble(&mut stiffness_matrix);
        }
        println!("stiffness_matrix = {:.3e}", stiffness_matrix);
    }
}
