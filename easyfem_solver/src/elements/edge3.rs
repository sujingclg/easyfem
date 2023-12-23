use std::ops::AddAssign;

use nalgebra::{DMatrix, DVector, Matrix3x1, MatrixXx2, MatrixXx3, Vector3};

use crate::{
    base::{gauss::get_gauss_1d_matrix, utils::flatten_vector},
    materials::Material,
};

use super::{GaussResult, GeneralElement, StructureElement};

pub struct Edge3 {
    node_dof: usize,                   // 节点自由度
    connectivity: [usize; 3],          // 单元的节点序号数组
    nodes_coordinates: Matrix3x1<f64>, // 单元节点的全局坐标数组, 每单元3节点, 每节点1坐标
    gauss_matrix: MatrixXx2<f64>,      // 高斯积分矩阵, 1列->w 2列->xi
    K: DMatrix<f64>,                   // 单元刚度矩阵
    F: DVector<f64>,                   // 右端向量
}

impl Edge3 {
    pub fn new(gauss_deg: usize, node_dof: usize) -> Self {
        Edge3 {
            node_dof,
            connectivity: [0; 3],
            nodes_coordinates: Matrix3x1::zeros(),
            gauss_matrix: get_gauss_1d_matrix(gauss_deg), // 高斯积分矩阵
            K: DMatrix::zeros(3 * node_dof, 3 * node_dof),
            F: DVector::zeros(3 * node_dof),
        }
    }

    fn gauss_point_calculate(&self, xi: f64) -> GaussResult<3, 1> {
        // 1维3节点等参元形函数
        let shp_val = Vector3::new(
            xi * (xi - 1.0) * 0.5, // N1
            xi * (xi + 1.0) * 0.5, // N2
            1.0 - xi.powi(2),      // N3
        );

        // 梯度矩阵，每个元素分别是形函数对xi求偏导: \frac{\partial Ni}{\partial \xi}
        let mut shp_grad = Matrix3x1::new(
            xi - 0.5,  // dN1/dxi
            xi + 0.5,  // dN2/dxi
            -2.0 * xi, // dN3/dxi
        );

        let mut dx_dxi = 0.0;

        // 遍历单元的每个节点
        for i in 0..self.connectivity.len() {
            dx_dxi += shp_grad[i] * self.nodes_coordinates[i];
        }
        // let det_J = f64::abs(dx_dxi);
        let det_J = dx_dxi; // jacob行列式

        shp_grad /= det_J; // 梯度矩阵除以jacob行列式以便进行单元组装

        GaussResult {
            shp_val,
            shp_grad,
            det_J,
        }
    }
}

impl GeneralElement<3, 1> for Edge3 {
    fn update(
        &mut self,
        element_number: usize,                   // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>,    // 全局单元-节点编号矩阵
        node_coordinate_matrix: &MatrixXx3<f64>, // 全局节点-坐标矩阵
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
        K: Fn(usize, usize, &GaussResult<3, 1>) -> DMatrix<f64>,
        F: Fn(usize, &GaussResult<3, 1>) -> DVector<f64>,
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

impl StructureElement<1> for Edge3 {
    // TODO: 此方法可以提取到和Edge2共用的Trait
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

    // TODO: 此方法可以提取到和Edge2共用的Trait
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
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 2, "edge3");
        let mut edge3 = Edge3::new(2, 1);
        let mut stiffness_matrix = DMatrix::zeros(5, 5);
        let mat = IsotropicLinearElastic1D::new(1.0, 1.0);
        for element_number in 0..mesh.get_element_count() {
            edge3.update(element_number, mesh.get_elements(), mesh.get_nodes());
            edge3.structure_stiffness_calculate(&mat);
            edge3.structure_assemble(&mut stiffness_matrix);
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
}
