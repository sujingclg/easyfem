use nalgebra::{DMatrix, Matrix2, Matrix2x1, MatrixXx2, MatrixXx3, Vector2};

use crate::{base::gauss::get_gauss_1d_matrix, materials::Material};

use super::{GeneralElement, StructureElement};

pub struct Edge2 {
    nodes_numbers: [usize; 2],         // 单元的节点序号数组
    nodes_coordinates: Matrix2x1<f64>, // 单元节点的全局坐标数组, 每单元2节点, 每节点1坐标
    gauss_matrix: MatrixXx2<f64>,      // 高斯积分矩阵, 1列->w 2列->xi
    K: Matrix2<f64>,                   // 单元刚度矩阵
                                       // F: Vector2<f64>,
}

impl Edge2 {
    pub fn new(gauss_deg: usize, // 高斯积分阶数
    ) -> Self {
        Edge2 {
            nodes_numbers: [0, 0],
            nodes_coordinates: Matrix2x1::zeros(),
            gauss_matrix: get_gauss_1d_matrix(gauss_deg),
            K: Matrix2::zeros(),
            // F: Vector2::zeros(),
        }
    }

    fn gauss_point_calculate(&self, xi: f64) -> (Vector2<f64>, Matrix2x1<f64>, f64) {
        // 1维2节点等参元形函数
        let shape_function_values = Vector2::new(
            (1.0 - xi) / 2.0, // N1
            (1.0 + xi) / 2.0, // N2
        );

        // 梯度矩阵，每个元素分别是形函数对xi求偏导: \frac{\partial Ni}{\partial \xi}
        let mut gradient_matrix = Matrix2x1::new(
            -0.5, // dN1/dxi
            0.5,  // dN2/dxi
        );

        let mut dx_dxi = 0.0;

        // 遍历单元的每个节点
        for i in 0..self.nodes_numbers.len() {
            dx_dxi += gradient_matrix[i] * self.nodes_coordinates[i];
        }
        // let det_jacob = f64::abs(dx_dxi);
        let det_J = dx_dxi; // jacob行列式

        gradient_matrix /= det_J; // 梯度矩阵除以jacob行列式以便进行单元组装

        (shape_function_values, gradient_matrix, det_J)
    }

    // pub fn general_calculate<K, F>(
    //     &mut self,
    //     F: &mut DVector<f64>,
    //     //  operator_K: K,
    //     operator_F: F,
    // ) where
    //     // K: Fn(f64) -> f64,
    //     F: Fn(usize, usize, &Vector2<f64>, &Matrix2x1<f64>) -> f64, // 输入分别为第i个节点的shape_function_value,
    // {
    //     //
    // }
}

impl GeneralElement for Edge2 {
    fn update(
        &mut self,
        element_number: usize,                   // 单元编号, 即单元的全局索引
        element_node_matrix: &DMatrix<usize>,    // 全局单元-节点编号矩阵, 每单元2节点
        node_coordinate_matrix: &MatrixXx3<f64>, // 全局节点-坐标矩阵, 每节点3坐标只取第一个
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
                self.nodes_coordinates.set_row(idx, &row.fixed_columns(0)); // 每节点3坐标只取第一个
            });
    }

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>) {
        for (i, node_i) in self.nodes_numbers.iter().enumerate() {
            for (j, node_j) in self.nodes_numbers.iter().enumerate() {
                stiffness_matrix[(*node_i, *node_j)] += self.K[(i, j)];
            }
        }
        self.K.fill(0.0);
    }
}

impl StructureElement<1> for Edge2 {
    fn structure_calculate(&mut self, mat: &impl Material<1>) {
        for row in self.gauss_matrix.row_iter() {
            let xi = row[1];
            let w = row[0];
            let (_, gradient_matrix, det_J) = self.gauss_point_calculate(xi);
            let JxW = det_J * w;
            for i in 0..self.nodes_numbers.len() {
                for j in 0..self.nodes_numbers.len() {
                    // 这里要对高斯积分进行累加
                    self.K[(i, j)] += gradient_matrix[j]
                        * gradient_matrix[i]
                        * mat.get_constitutive_matrix()[(0, 0)]
                        * JxW;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::materials::IsotropicLinearElastic1D;

    use super::*;

    #[test]
    fn structure_test_1() {
        let element_node_matrix = DMatrix::from_row_slice(5, 2, &[0, 1, 1, 2, 2, 3, 3, 4, 4, 5]);
        // println!("{}", element_node_matrix);
        let node_coordinate_matrix = MatrixXx3::from_row_slice(&[
            0.0, 0.0, 0.0, // 0
            0.2, 0.0, 0.0, // 1
            0.4, 0.0, 0.0, // 2
            0.6, 0.0, 0.0, // 3
            0.8, 0.0, 0.0, // 4
            1.0, 0.0, 0.0, // 5
        ]);
        let mut edge2 = Edge2::new(2);
        let mut stiffness_matrix = DMatrix::zeros(6, 6);
        // let D = Matrix1::new(1.0e9);
        let mat = IsotropicLinearElastic1D::new(1.0e9, 1.0);
        for element_number in 0..element_node_matrix.nrows() {
            edge2.update(
                element_number,
                &element_node_matrix,
                &node_coordinate_matrix,
            );
            edge2.structure_calculate(&mat);
            edge2.assemble(&mut stiffness_matrix);
        }
        println!("{:.3e}", stiffness_matrix);
    }

    #[test]
    fn poisson_test_1() {
        //
    }
}
