use crate::{
    base::{
        gauss::{Gauss, GaussEdge2, GaussEdge3},
        primitives::{Edge, GeneralElement, PrimitiveBase},
    },
    materials::Material,
};

use super::StructureElement;

pub struct StructureEdge<const N: usize> {
    edge: Edge<N>,
    gauss: Box<dyn Gauss<N, 1>>,
}

pub type StructureEdge2 = StructureEdge<2>;
impl StructureEdge2 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        StructureEdge2 {
            edge: Edge::<2>::new(node_dof),
            gauss: Box::new(GaussEdge2::new(gauss_deg)),
        }
    }
}

pub type StructureEdge3 = StructureEdge<3>;
impl StructureEdge3 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        StructureEdge3 {
            edge: Edge::<3>::new(node_dof),
            gauss: Box::new(GaussEdge3::new(gauss_deg)),
        }
    }
}

impl<const N: usize> StructureElement<1> for StructureEdge<N> {
    fn update(
        &mut self,
        element_number: usize, // 单元编号, 即单元的全局索引
        connectivity_matrix: &nalgebra::DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &nalgebra::MatrixXx3<f64>, // 全局节点-坐标矩阵
    ) {
        self.edge
            .update(element_number, connectivity_matrix, coordinate_matrix);
    }

    fn structure_stiffness_calc(&mut self, mat: &impl Material<1>) {
        let node_count = self.edge.node_count();
        for (w, gauss_point) in self.gauss.gauss_vector() {
            let gauss_result = self
                .gauss
                .shape_func_calc(self.edge.nodes_coordinates(), gauss_point);
            let JxW = gauss_result.det_j * w;
            for i in 0..node_count {
                for j in 0..node_count {
                    // 这里要对高斯积分进行累加
                    self.edge.K_mut()[(i, j)] += gauss_result.shp_grad[j]
                        * gauss_result.shp_grad[i]
                        * mat.get_constitutive_matrix()[(0, 0)]
                        * JxW;
                }
            }
        }
    }

    fn assemble(
        &mut self,
        stiffness_matrix: &mut nalgebra::DMatrix<f64>,
        right_vector: &mut nalgebra::DVector<f64>,
    ) {
        self.edge.assemble(stiffness_matrix, right_vector);
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange1DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, SMatrix};

    use crate::materials::IsotropicLinearElastic1D;

    use super::*;

    #[test]
    fn structure_edge2_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 5, "edge2");
        let mut edge2 = StructureEdge2::new(1, 2);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic1D::new(1.0e9, 1.0);

        for element_number in 0..mesh.element_count() {
            edge2.update(element_number, mesh.elements(), mesh.nodes());
            edge2.structure_stiffness_calc(&mat);
            edge2.assemble(&mut stiffness_matrix, &mut right_vector);
        }
        println!("{:.3e}", stiffness_matrix);
        let answer = SMatrix::<f64, 6, 6>::new(
            5.000e9, -5.000e9, 0.000e0, 0.000e0, 0.000e0, 0.000e0, -5.000e9, 1.000e10, -5.000e9,
            0.000e0, 0.000e0, 0.000e0, 0.000e0, -5.000e9, 1.000e10, -5.000e9, 0.000e0, 0.000e0,
            0.000e0, 0.000e0, -5.000e9, 1.000e10, -5.000e9, 0.000e0, 0.000e0, 0.000e0, 0.000e0,
            -5.000e9, 1.000e10, -5.000e9, 0.000e0, 0.000e0, 0.000e0, 0.000e0, -5.000e9, 5.000e9,
        );
        let err = 1e-15;
        assert!(stiffness_matrix.relative_eq(&answer, err, err));
    }

    #[test]
    fn structure_edge3_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 2, "edge3");
        let mut edge3 = StructureEdge3::new(1, 2);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic1D::new(1.0, 1.0);

        for element_number in 0..mesh.element_count() {
            edge3.update(element_number, mesh.elements(), mesh.nodes());
            edge3.structure_stiffness_calc(&mat);
            edge3.assemble(&mut stiffness_matrix, &mut right_vector);
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
