use nalgebra::{Matrix2x3, Matrix3x2};

use crate::{
    base::{
        elements::{ElementBase, Quad},
        gauss::{Gauss, GaussResult},
    },
    materials::Material,
};

use super::StructureElement;

impl<const N: usize> StructureElement<3, N, 2> for Quad<N> {
    fn structure_stiffness_calc(&mut self, gauss: &impl Gauss<N, 2>, mat: &impl Material<3>) {
        let node_count = self.node_count();

        let mut B = Matrix3x2::zeros(); // 应变矩阵
        let mut Bt = Matrix2x3::zeros(); // 应变矩阵的转置
        for (w, gauss_point) in gauss.gauss_vector() {
            // let xi = row[1];
            // let eta = row[2];
            // let w = row[0];
            let GaussResult {
                shp_grad, det_j, ..
            } = gauss.shape_func_calc(self.nodes_coordinates(), gauss_point);
            let JxW = det_j * w;
            for i in 0..node_count {
                B[(0, 0)] = shp_grad[(i, 0)]; // 矩阵分块乘法, 每次计算出2x2的矩阵, 然后组装到单元刚度矩阵的对应位置
                B[(1, 1)] = shp_grad[(i, 1)];
                B[(2, 0)] = shp_grad[(i, 1)];
                B[(2, 1)] = shp_grad[(i, 0)];
                for j in 0..node_count {
                    Bt[(0, 0)] = shp_grad[(j, 0)];
                    Bt[(0, 2)] = shp_grad[(j, 1)];
                    Bt[(1, 1)] = shp_grad[(j, 1)];
                    Bt[(1, 2)] = shp_grad[(j, 0)];
                    let C = Bt * mat.get_constitutive_matrix() * B;
                    // 这里要对高斯积分进行累加
                    // for (r, c) in square_range(2) {
                    //     self.K_mut()[(2 * i + r, 2 * j + c)] += C[(r, c)] * JxW;
                    // }
                    self.K_mut()[(2 * i + 0, 2 * j + 0)] += C[(0, 0)] * JxW; // K_ux,ux
                    self.K_mut()[(2 * i + 0, 2 * j + 1)] += C[(0, 1)] * JxW; // K_ux,uy
                    self.K_mut()[(2 * i + 1, 2 * j + 0)] += C[(1, 0)] * JxW; // K_uy,ux
                    self.K_mut()[(2 * i + 1, 2 * j + 1)] += C[(1, 1)] * JxW;
                    // K_uy,uy
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange2DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, MatrixXx3, SMatrix};

    use crate::{
        base::{
            elements::{GeneralElement, Quad4, Quad9},
            gauss::{GaussQuad4, GaussQuad9},
        },
        materials::{IsotropicLinearElastic2D, PlaneCondition::*},
    };

    use super::*;

    #[test]
    /// Daryl L. Logan 有限元方法基础教程(第五版)  例10.4
    fn structure_quad4_test() {
        let mesh = Lagrange2DMesh::new(3.0, 5.0, 1, 2.0, 4.0, 1, "quad4");
        println!("{}", mesh);
        let n_dofs = mesh.node_count() * 2;
        let mut quad4 = Quad4::new(2);
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic2D::new(3.0e7, 0.25, PlaneStress, 1.0);
        let gauss = GaussQuad4::new(2);

        for element_number in 0..mesh.element_count() {
            quad4.update(element_number, mesh.elements(), mesh.nodes());
            quad4.structure_stiffness_calc(&gauss, &mat);
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
        let mut quad9 = Quad9::new(2);
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic2D::new(3.0e7, 0.25, PlaneStress, 1.0);
        // let gauss = Gauss::Quad(GaussQuad::new(2));
        let gauss = GaussQuad9::new(2);
        for element_number in 0..connectivity_matrix.nrows() {
            quad9.update(
                element_number,
                &connectivity_matrix,
                &node_coordinate_matrix,
            );
            quad9.structure_stiffness_calc(&gauss, &mat);
            quad9.assemble(&mut stiffness_matrix, &mut right_vector);
        }
        println!("stiffness_matrix = {:.3e}", stiffness_matrix);
    }
}
