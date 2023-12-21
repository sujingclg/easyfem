use nalgebra::{DMatrix, Matrix2x3, Matrix3x2, Matrix4x2, MatrixXx3, SMatrix, Vector4};

use crate::{base::gauss::get_gauss_2d_matrix, materials::Material};

use super::{GeneralElement, StructureElement};

pub struct Quad4 {
    nodes_numbers: [usize; 4],         // 单元的节点序号数组
    nodes_coordinates: Matrix4x2<f64>, // 单元节点的全局坐标数组, 每单元4节点, 每节点2坐标
    gauss_matrix: MatrixXx3<f64>,      // 高斯积分矩阵, 1列->w 2列->xi 3列->eta
    K: SMatrix<f64, 8, 8>,             // 单元刚度矩阵
}

impl Quad4 {
    pub fn new(gauss_deg: usize, // 高斯积分阶数
    ) -> Self {
        Quad4 {
            nodes_numbers: [0; 4],
            nodes_coordinates: Matrix4x2::zeros(),
            gauss_matrix: get_gauss_2d_matrix(gauss_deg),
            K: SMatrix::zeros(),
        }
    }

    // 在每个高斯点上做个预计算
    fn gauss_point_calculate(&self, xi: f64, eta: f64) -> (Vector4<f64>, Matrix4x2<f64>, f64) {
        // 2维4节点等参元形函数
        let shape_function_values = Vector4::new(
            (1.0 - xi) * (1.0 - eta) / 4.0, // N1
            (1.0 + xi) * (1.0 - eta) / 4.0, // N2
            (1.0 + xi) * (1.0 + eta) / 4.0, // N3
            (1.0 - xi) * (1.0 + eta) / 4.0, // N4
        );

        // 梯度矩阵，每个元素分别是形函数分别对xi和eta求偏导: \frac{\partial Ni}{\partial \xi} \frac{\partial Ni}{\partial \eta}
        let mut gradient_matrix = Matrix4x2::new(
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
        let J = gradient_matrix.transpose() * self.nodes_coordinates;
        let det_J = J.determinant(); // 行列式

        // jacob逆矩阵形式为[[dxi_dx, deta_dx],[dxi_dy, deta_dy]]
        let inverse_J = J.try_inverse().unwrap(); // 逆矩阵

        // gradient_matrix 的每一行变为下式, Ni从N1到N4
        // dNi/dxi * dxi/dx + dNi/deta * deta/dx, dNi/dxi * dxi/dy + dNi/deta * deta/dy
        gradient_matrix = gradient_matrix * inverse_J.transpose();

        (shape_function_values, gradient_matrix, det_J)
    }
}

impl GeneralElement for Quad4 {
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
        for (i, node_i) in self.nodes_numbers.iter().enumerate() {
            for (j, node_j) in self.nodes_numbers.iter().enumerate() {
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

impl StructureElement<3> for Quad4 {
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
    use easyfem_mesh::{Lagrange2DMesh, Mesh};

    use crate::materials::{IsotropicLinearElastic2D, PlaneCondition};

    use super::*;

    #[test]
    fn structure_test_1() {
        let n_dofs: usize = 8;
        let element_node_matrix = DMatrix::from_row_slice(1, 4, &[0, 1, 2, 3]);
        let node_coordinate_matrix = MatrixXx3::from_row_slice(&[
            3.0, 2.0, 0.0, // 0
            5.0, 2.0, 0.0, // 1
            5.0, 4.0, 0.0, // 2
            3.0, 4.0, 0.0, // 3
        ]);
        let mut quad4 = Quad4::new(2);
        let mat = IsotropicLinearElastic2D::new(3.0e7, 0.25, PlaneCondition::PlaneStress, 1.0);
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
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

    #[test]
    fn structure_test_2() {
        let n_dofs: usize = 8;
        let mesh = Lagrange2DMesh::new(3.0, 5.0, 1, 2.0, 4.0, 1, "quad4");
        println!("{}", mesh);
        let mut quad4 = Quad4::new(2);
        let mat = IsotropicLinearElastic2D::new(3.0e7, 0.25, PlaneCondition::PlaneStress, 1.0);
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        for element_number in 0..mesh.get_element_count() {
            quad4.update(element_number, mesh.get_elements(), mesh.get_nodes());
            quad4.structure_calculate(&mat);
            quad4.assemble(&mut stiffness_matrix);
        }
        println!("stiffness_matrix = {:.3e}", stiffness_matrix);
    }
}
