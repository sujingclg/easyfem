use nalgebra::{DMatrix, Matrix3x6, Matrix6x3, MatrixXx3, MatrixXx4, SMatrix};

use crate::{base::gauss::get_gauss_3d_matrix, materials::Material};

use super::{GeneralElement, StructureElement};

type Matrix24<T> = SMatrix<T, 24, 24>;
type Matrix8x3<T> = SMatrix<T, 8, 3>;

pub struct Solid8 {
    nodes_numbers: [usize; 8],         // 单元的节点序号数组
    nodes_coordinates: Matrix8x3<f64>, // 单元节点的全局坐标数组, 每单元8节点, 每节点3坐标
    gauss_matrix: MatrixXx4<f64>,      // 高斯积分矩阵, 1列->w 2列->xi 3列->eta 4列->zeta
    K: Matrix24<f64>,                  // 单元刚度矩阵
}

impl Solid8 {
    pub fn new(gauss_deg: usize) -> Self {
        Solid8 {
            nodes_numbers: [0, 0, 0, 0, 0, 0, 0, 0],
            nodes_coordinates: Matrix8x3::<f64>::zeros(),
            gauss_matrix: get_gauss_3d_matrix(gauss_deg),
            K: Matrix24::<f64>::zeros(),
        }
    }

    // 在每个高斯点上做个预计算
    fn gauss_point_calculate(
        &self,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> (SMatrix<f64, 8, 1>, Matrix8x3<f64>, f64) {
        // 3维8节点六面体等参元形函数
        let shape_function_values = SMatrix::<f64, 8, 1>::from_column_slice(&[
            (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) / 8.0, // N1
            (1.0 + xi) * (1.0 - eta) * (1.0 - zeta) / 8.0, // N2
            (1.0 + xi) * (1.0 + eta) * (1.0 - zeta) / 8.0, // N3
            (1.0 - xi) * (1.0 + eta) * (1.0 - zeta) / 8.0, // N4
            (1.0 - xi) * (1.0 - eta) * (1.0 + zeta) / 8.0, // N5
            (1.0 + xi) * (1.0 - eta) * (1.0 + zeta) / 8.0, // N6
            (1.0 + xi) * (1.0 + eta) * (1.0 + zeta) / 8.0, // N7
            (1.0 - xi) * (1.0 + eta) * (1.0 + zeta) / 8.0, // N8
        ]);

        /*
         * 梯度矩阵，每行元素是每个形函数分别对xi、eta、zeta求偏导:
         *  第一列 \frac{\partial Ni}{\partial \xi}
         *  第二列 \frac{\partial Ni}{\partial \eta}
         *  第三列 \frac{\partial Ni}{\partial \zeta}
         */
        let mut gradient_matrix = Matrix8x3::from_row_slice(&[
            // N1
            -(1.0 - eta) * (1.0 - zeta) / 8.0, // dN1/dxi
            -(1.0 - xi) * (1.0 - zeta) / 8.0,  // dN1/deta
            -(1.0 - xi) * (1.0 - eta) / 8.0,   // dN1/dzeta
            // N2
            (1.0 - eta) * (1.0 - zeta) / 8.0, // dN2/dxi
            -(1.0 + xi) * (1.0 - zeta) / 8.0, // dN2/deta
            -(1.0 + xi) * (1.0 - eta) / 8.0,  // dN2/dzeta
            // N3
            (1.0 + eta) * (1.0 - zeta) / 8.0, // dN3/dxi
            (1.0 + xi) * (1.0 - zeta) / 8.0,  // dN3/deta
            -(1.0 + xi) * (1.0 + eta) / 8.0,  // dN3/dzeta
            // N4
            -(1.0 + eta) * (1.0 - zeta) / 8.0, // dN4/xi
            (1.0 - xi) * (1.0 - zeta) / 8.0,   // dN4/deta
            -(1.0 - xi) * (1.0 + eta) / 8.0,   // dN4/dzeta
            // N5
            -(1.0 - eta) * (1.0 + zeta) / 8.0, // dN5/dxi
            -(1.0 - xi) * (1.0 + zeta) / 8.0,  // dN5/deta
            (1.0 - xi) * (1.0 - eta) / 8.0,    // dN5/dzeta
            // N6
            (1.0 - eta) * (1.0 + zeta) / 8.0, // dN6/dxi
            -(1.0 + xi) * (1.0 + zeta) / 8.0, // dN6/deta
            (1.0 + xi) * (1.0 - eta) / 8.0,   // dN6/dzeta
            // N7
            (1.0 + eta) * (1.0 + zeta) / 8.0, // dN7/dxi
            (1.0 + xi) * (1.0 + zeta) / 8.0,  // dN7/deta
            (1.0 + xi) * (1.0 + eta) / 8.0,   // dN7/dzeta
            // N8
            -(1.0 + eta) * (1.0 + zeta) / 8.0, // dN8/dxi
            (1.0 - xi) * (1.0 + zeta) / 8.0,   // dN8/deta
            (1.0 - xi) * (1.0 + eta) / 8.0,    // dN8/dzeta
        ]);

        /*
         * jacob矩阵形式为
         * [[dx_dxi, dy_dxi, dz_dxi],
         * [dx_deta, dy_deta, dz_deta],
         * [dx_dzeta, dy_dzeta, dz_dzeta]]
         */
        let J = gradient_matrix.transpose() * self.nodes_coordinates;
        let det_J = J.determinant();

        /*
         * jacob逆矩阵形式为
         * [[dxi_dx, deta_dx, dzeta_dx],
         * [dxi_dy, deta_dy, dzeta_dy],
         * [dxi_dz, deta_dz, dzeta_dz]]
         */
        let inverse_J = J.try_inverse().unwrap(); // 逆矩阵

        gradient_matrix = gradient_matrix * inverse_J.transpose();

        (shape_function_values, gradient_matrix, det_J)
    }
}

impl GeneralElement for Solid8 {
    // TODO: 将此方法集成到GeneralElement Trait中
    fn update(
        &mut self,
        element_number: usize,                   // 单元编号, 即单元的全局索引
        element_node_matrix: &DMatrix<usize>,    // 全局单元-节点编号矩阵, 每单元8节点
        node_coordinate_matrix: &MatrixXx3<f64>, // 全局节点-坐标矩阵, 每节点3坐标
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
                self.nodes_coordinates.set_row(idx, &row.fixed_columns(0));
            });
    }

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>) {
        for (i, node_i) in self.nodes_numbers.iter().enumerate() {
            for (j, node_j) in self.nodes_numbers.iter().enumerate() {
                stiffness_matrix[(3 * node_i + 0, 3 * node_j + 0)] +=
                    self.K[(3 * i + 0, 3 * j + 0)];
                stiffness_matrix[(3 * node_i + 0, 3 * node_j + 1)] +=
                    self.K[(3 * i + 0, 3 * j + 1)];
                stiffness_matrix[(3 * node_i + 0, 3 * node_j + 2)] +=
                    self.K[(3 * i + 0, 3 * j + 2)];
                stiffness_matrix[(3 * node_i + 1, 3 * node_j + 0)] +=
                    self.K[(3 * i + 1, 3 * j + 0)];
                stiffness_matrix[(3 * node_i + 1, 3 * node_j + 1)] +=
                    self.K[(3 * i + 1, 3 * j + 1)];
                stiffness_matrix[(3 * node_i + 1, 3 * node_j + 2)] +=
                    self.K[(3 * i + 1, 3 * j + 2)];
                stiffness_matrix[(3 * node_i + 2, 3 * node_j + 0)] +=
                    self.K[(3 * i + 2, 3 * j + 0)];
                stiffness_matrix[(3 * node_i + 2, 3 * node_j + 1)] +=
                    self.K[(3 * i + 2, 3 * j + 1)];
                stiffness_matrix[(3 * node_i + 2, 3 * node_j + 2)] +=
                    self.K[(3 * i + 2, 3 * j + 2)];
            }
        }
        self.K.fill(0.0);
    }
}

impl StructureElement<6> for Solid8 {
    fn structure_calculate(&mut self, mat: &impl Material<6>) {
        let mut B = Matrix6x3::<f64>::zeros(); // 应变矩阵
        let mut Bt = Matrix3x6::<f64>::zeros(); // 应变矩阵的转置
        for row in self.gauss_matrix.row_iter() {
            let xi = row[1];
            let eta = row[2];
            let zeta = row[3];
            let w = row[0];
            let (_, gradient_matrix, det_J) = self.gauss_point_calculate(xi, eta, zeta);
            let JxW = det_J * w;
            for i in 0..self.nodes_numbers.len() {
                B[(0, 0)] = gradient_matrix[(i, 0)]; // 矩阵分块乘法, 每次计算出3x3的矩阵, 然后组装到单元刚度矩阵的对应位置
                B[(1, 1)] = gradient_matrix[(i, 1)];
                B[(2, 2)] = gradient_matrix[(i, 2)];
                B[(3, 0)] = gradient_matrix[(i, 1)];
                B[(3, 1)] = gradient_matrix[(i, 0)];
                B[(4, 1)] = gradient_matrix[(i, 2)];
                B[(4, 2)] = gradient_matrix[(i, 1)];
                B[(5, 0)] = gradient_matrix[(i, 2)];
                B[(5, 2)] = gradient_matrix[(i, 0)];
                for j in 0..self.nodes_numbers.len() {
                    Bt[(0, 0)] = gradient_matrix[(j, 0)];
                    Bt[(0, 3)] = gradient_matrix[(j, 1)];
                    Bt[(0, 5)] = gradient_matrix[(j, 2)];
                    Bt[(1, 1)] = gradient_matrix[(j, 1)];
                    Bt[(1, 3)] = gradient_matrix[(j, 0)];
                    Bt[(1, 4)] = gradient_matrix[(j, 2)];
                    Bt[(2, 2)] = gradient_matrix[(j, 2)];
                    Bt[(2, 4)] = gradient_matrix[(j, 1)];
                    Bt[(2, 5)] = gradient_matrix[(j, 0)];
                    let C = Bt * mat.get_constitutive_matrix() * B;
                    // 这里要对高斯积分进行累加
                    self.K[(3 * i + 0, 3 * j + 0)] += C[(0, 0)] * JxW; // K_ux,ux
                    self.K[(3 * i + 0, 3 * j + 1)] += C[(0, 1)] * JxW; // K_ux,uy
                    self.K[(3 * i + 0, 3 * j + 2)] += C[(0, 2)] * JxW; // K_ux,uz
                    self.K[(3 * i + 1, 3 * j + 0)] += C[(1, 0)] * JxW; // K_uy,ux
                    self.K[(3 * i + 1, 3 * j + 1)] += C[(1, 1)] * JxW; // K_uy,uy
                    self.K[(3 * i + 1, 3 * j + 2)] += C[(1, 2)] * JxW; // K_uy,uz
                    self.K[(3 * i + 2, 3 * j + 0)] += C[(2, 0)] * JxW; // K_uz,ux
                    self.K[(3 * i + 2, 3 * j + 1)] += C[(2, 1)] * JxW; // K_uz,uy
                    self.K[(3 * i + 2, 3 * j + 2)] += C[(2, 2)] * JxW; // K_uz,uz
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use crate::materials::IsotropicLinearElastic3D;

    use super::*;

    #[test]
    fn structure_test_1() {
        let n_dofs: usize = 24;
        let element_node_matrix = DMatrix::from_row_slice(1, 8, &[0, 1, 2, 3, 4, 5, 6, 7]);
        let node_coordinate_matrix = MatrixXx3::from_row_slice(&[
            0.2, 0.0, 0.0, // 0
            0.2, 0.8, 0.0, // 1
            0.0, 0.8, 0.0, // 2
            0.0, 0.0, 0.0, // 3
            0.2, 0.0, 0.6, // 4
            0.2, 0.8, 0.6, // 5
            0.0, 0.8, 0.6, // 6
            0.0, 0.0, 0.6, // 7
        ]);
        let mut solid8 = Solid8::new(2);
        let mat = IsotropicLinearElastic3D::new(1.0e10, 0.25);
        let mut stiffness_matrix = DMatrix::<f64>::zeros(n_dofs, n_dofs);
        for element_number in 0..element_node_matrix.nrows() {
            solid8.update(
                element_number,
                &element_node_matrix,
                &node_coordinate_matrix,
            );
            solid8.structure_calculate(&mat);
            solid8.assemble(&mut stiffness_matrix);
        }
        let penalty = 1.0e20;
        let mut force_vector = DVector::zeros(n_dofs);
        force_vector[5 * 3 + 2] = -1.0e5;
        force_vector[6 * 3 + 2] = -1.0e5;
        // println!("stiffness_matrix = {:.1e}", stiffness_matrix);
        for i in [0, 3, 7, 4] as [usize; 4] {
            let x = i * 3 + 0;
            let y = i * 3 + 1;
            let z = i * 3 + 2;
            stiffness_matrix[(x, x)] *= penalty;
            stiffness_matrix[(y, y)] *= penalty;
            stiffness_matrix[(z, z)] *= penalty;
        }
        // println!("stiffness_matrix = {:.1e}", &stiffness_matrix);
        // println!("{:.3e}", &force_vector);
        let displacement_vector = stiffness_matrix.lu().solve(&force_vector);
        if let Some(d) = displacement_vector {
            println!("{:.4}", d * 1.0e3);
        }
    }
}
