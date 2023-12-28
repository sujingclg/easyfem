use nalgebra::{Matrix3x6, Matrix6x3};

use crate::{
    base::{
        gauss::{Gauss, GaussCube8, GaussResult},
        primitives::{Cube, GeneralElement, PrimitiveBase},
    },
    materials::Material,
};

use super::StructureElement;

pub struct StructureCube<const N: usize> {
    cube: Cube<N>,
    gauss: Box<dyn Gauss<N, 3>>,
}

pub type StructureCube8 = StructureCube<8>;
impl StructureCube8 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        StructureCube8 {
            cube: Cube::<8>::new(node_dof),
            gauss: Box::new(GaussCube8::new(gauss_deg)),
        }
    }
}

impl<const N: usize> StructureElement<6> for StructureCube<N> {
    fn update(
        &mut self,
        element_number: usize, // 单元编号, 即单元的全局索引
        connectivity_matrix: &nalgebra::DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &nalgebra::MatrixXx3<f64>, // 全局节点-坐标矩阵
    ) {
        self.cube
            .update(element_number, connectivity_matrix, coordinate_matrix);
    }

    fn structure_stiffness_calc(&mut self, mat: &impl Material<6>) {
        let node_count = self.cube.node_count();

        let mut B = Matrix6x3::zeros(); // 应变矩阵
        let mut Bt = Matrix3x6::zeros(); // 应变矩阵的转置
        for (w, gauss_point) in self.gauss.gauss_vector() {
            let GaussResult {
                shp_grad, det_j, ..
            } = self
                .gauss
                .shape_func_calc(self.cube.nodes_coordinates(), gauss_point);
            let JxW = det_j * w;
            for i in 0..node_count {
                B[(0, 0)] = shp_grad[(i, 0)]; // 矩阵分块乘法, 每次计算出3x3的矩阵, 然后组装到单元刚度矩阵的对应位置
                B[(1, 1)] = shp_grad[(i, 1)];
                B[(2, 2)] = shp_grad[(i, 2)];
                B[(3, 0)] = shp_grad[(i, 1)];
                B[(3, 1)] = shp_grad[(i, 0)];
                B[(4, 1)] = shp_grad[(i, 2)];
                B[(4, 2)] = shp_grad[(i, 1)];
                B[(5, 0)] = shp_grad[(i, 2)];
                B[(5, 2)] = shp_grad[(i, 0)];
                for j in 0..node_count {
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
                    self.cube.K_mut()[(3 * i + 0, 3 * j + 0)] += C[(0, 0)] * JxW; // K_ux,ux
                    self.cube.K_mut()[(3 * i + 0, 3 * j + 1)] += C[(0, 1)] * JxW; // K_ux,uy
                    self.cube.K_mut()[(3 * i + 0, 3 * j + 2)] += C[(0, 2)] * JxW; // K_ux,uz
                    self.cube.K_mut()[(3 * i + 1, 3 * j + 0)] += C[(1, 0)] * JxW; // K_uy,ux
                    self.cube.K_mut()[(3 * i + 1, 3 * j + 1)] += C[(1, 1)] * JxW; // K_uy,uy
                    self.cube.K_mut()[(3 * i + 1, 3 * j + 2)] += C[(1, 2)] * JxW; // K_uy,uz
                    self.cube.K_mut()[(3 * i + 2, 3 * j + 0)] += C[(2, 0)] * JxW; // K_uz,ux
                    self.cube.K_mut()[(3 * i + 2, 3 * j + 1)] += C[(2, 1)] * JxW; // K_uz,uy
                    self.cube.K_mut()[(3 * i + 2, 3 * j + 2)] += C[(2, 2)] * JxW;
                    // K_uz,uz
                }
            }
        }
    }

    fn assemble(
        &mut self,
        stiffness_matrix: &mut nalgebra::DMatrix<f64>,
        right_vector: &mut nalgebra::DVector<f64>,
    ) {
        self.cube.assemble(stiffness_matrix, right_vector);
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange3DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, SMatrix};

    use crate::materials::IsotropicLinearElastic3D;

    use super::*;

    #[test]
    /// 曾攀 有限元分析基础教程 算例4.8.2
    fn structure_cube8_test() {
        let n_dofs: usize = 24;
        let mesh = Lagrange3DMesh::new(0.0, 0.2, 1, 0.0, 0.8, 1, 0.0, 0.6, 1, "cube8");
        // let mut cube8 = Cube8::new(3);
        let mut cube8 = StructureCube8::new(3, 2);
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic3D::new(1.0e10, 0.25);

        for element_number in 0..mesh.element_count() {
            cube8.update(element_number, mesh.elements(), mesh.nodes());
            cube8.structure_stiffness_calc(&mat);
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
        let err = 1e-4;
        if let Some(d) = displacement_vector {
            assert!((&d * 1.0e3).relative_eq(&answer, err, err));
            println!("{:.4}", &d * 1.0e3);
        }
    }
}
