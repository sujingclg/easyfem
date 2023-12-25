use nalgebra::{Matrix3x6, Matrix6x3};

use crate::{
    base::{
        elements::{Cube8, ElementBase},
        gauss::{Gauss, GaussResult},
    },
    materials::Material,
};

use super::StructureElement;

impl StructureElement<6, 8, 3> for Cube8 {
    fn structure_stiffness_calculate(&mut self, mat: &impl Material<6>, gauss: &Gauss) {
        if let Gauss::Cube(gauss_cube) = gauss {
            let mut B = Matrix6x3::zeros(); // 应变矩阵
            let mut Bt = Matrix3x6::zeros(); // 应变矩阵的转置
            for row in gauss_cube.get_gauss_matrix().row_iter() {
                let xi = row[1];
                let eta = row[2];
                let zeta = row[3];
                let w = row[0];
                let GaussResult {
                    shp_grad, det_j, ..
                } = gauss_cube.linear_shape_func_calc(&self.nodes_coordinates(), [xi, eta, zeta]);
                let JxW = det_j * w;
                for i in 0..self.connectivity().len() {
                    B[(0, 0)] = shp_grad[(i, 0)]; // 矩阵分块乘法, 每次计算出3x3的矩阵, 然后组装到单元刚度矩阵的对应位置
                    B[(1, 1)] = shp_grad[(i, 1)];
                    B[(2, 2)] = shp_grad[(i, 2)];
                    B[(3, 0)] = shp_grad[(i, 1)];
                    B[(3, 1)] = shp_grad[(i, 0)];
                    B[(4, 1)] = shp_grad[(i, 2)];
                    B[(4, 2)] = shp_grad[(i, 1)];
                    B[(5, 0)] = shp_grad[(i, 2)];
                    B[(5, 2)] = shp_grad[(i, 0)];
                    for j in 0..self.connectivity().len() {
                        Bt[(0, 0)] = shp_grad[(j, 0)];
                        Bt[(0, 3)] = shp_grad[(j, 1)];
                        Bt[(0, 5)] = shp_grad[(j, 2)];
                        Bt[(1, 1)] = shp_grad[(j, 1)];
                        Bt[(1, 3)] = shp_grad[(j, 0)];
                        Bt[(1, 4)] = shp_grad[(j, 2)];
                        Bt[(2, 2)] = shp_grad[(j, 2)];
                        Bt[(2, 4)] = shp_grad[(j, 1)];
                        Bt[(2, 5)] = shp_grad[(j, 0)];
                        let C = Bt * mat.get_constitutive_matrix() * B;
                        // 这里要对高斯积分进行累加
                        self.K_mut()[(3 * i + 0, 3 * j + 0)] += C[(0, 0)] * JxW; // K_ux,ux
                        self.K_mut()[(3 * i + 0, 3 * j + 1)] += C[(0, 1)] * JxW; // K_ux,uy
                        self.K_mut()[(3 * i + 0, 3 * j + 2)] += C[(0, 2)] * JxW; // K_ux,uz
                        self.K_mut()[(3 * i + 1, 3 * j + 0)] += C[(1, 0)] * JxW; // K_uy,ux
                        self.K_mut()[(3 * i + 1, 3 * j + 1)] += C[(1, 1)] * JxW; // K_uy,uy
                        self.K_mut()[(3 * i + 1, 3 * j + 2)] += C[(1, 2)] * JxW; // K_uy,uz
                        self.K_mut()[(3 * i + 2, 3 * j + 0)] += C[(2, 0)] * JxW; // K_uz,ux
                        self.K_mut()[(3 * i + 2, 3 * j + 1)] += C[(2, 1)] * JxW; // K_uz,uy
                        self.K_mut()[(3 * i + 2, 3 * j + 2)] += C[(2, 2)] * JxW;
                        // K_uz,uz
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange3DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, SMatrix};

    use crate::{
        base::{elements::GeneralElement, gauss::GaussCube},
        materials::IsotropicLinearElastic3D,
    };

    use super::*;

    #[test]
    /// 曾攀 有限元分析基础教程 算例4.8.2
    fn structure_cube8_test() {
        let n_dofs: usize = 24;
        let mesh = Lagrange3DMesh::new(0.0, 0.2, 1, 0.0, 0.8, 1, 0.0, 0.6, 1, "cube8");
        let mut cube8 = Cube8::new(3);
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic3D::new(1.0e10, 0.25);
        let gauss = Gauss::Cube(GaussCube::new(2));
        for element_number in 0..mesh.get_element_count() {
            cube8.update(element_number, mesh.get_elements(), mesh.get_nodes());
            cube8.structure_stiffness_calculate(&mat, &gauss);
            cube8.assemble(&mut stiffness_matrix, &mut right_vector);
        }
        let penalty = 1.0e20;
        let mut force_vector = DVector::zeros(n_dofs);
        force_vector[6 * 3 + 2] = -1.0e5;
        force_vector[7 * 3 + 2] = -1.0e5;
        if let Some(left_nodes) = mesh.get_boundary_node_ids().get("left") {
            for i in left_nodes {
                let x = i * 3 + 0;
                let y = i * 3 + 1;
                let z = i * 3 + 2;
                stiffness_matrix[(x, x)] *= penalty;
                stiffness_matrix[(y, y)] *= penalty;
                stiffness_matrix[(z, z)] *= penalty;
            }
        }

        let displacement_vector = stiffness_matrix.lu().solve(&force_vector);
        let answer = SMatrix::<f64, 24, 1>::from_row_slice(&[
            -0.0000, -0.0000, -0.0000, 0.0000, -0.0000, -0.0000, -0.0223, -0.2769, -0.6728, 0.0223,
            -0.2769, -0.6728, 0.0000, 0.0000, -0.0000, -0.0000, 0.0000, -0.0000, 0.0129, 0.3108,
            -0.7774, -0.0129, 0.3108, -0.7774,
        ]);
        let err = 1e-3;
        if let Some(d) = displacement_vector {
            assert!((&d * 1.0e3).relative_eq(&answer, err, err));
            println!("{:.4}", &d * 1.0e3);
        }
    }
}
