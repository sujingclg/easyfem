use crate::{
    base::{
        elements::{Edge2, Edge3, ElementBase},
        gauss::Gauss,
    },
    materials::Material,
};

use super::StructureElement;

impl StructureElement<1, 2> for Edge2 {
    fn structure_stiffness_calc(&mut self, gauss: &Gauss, mat: &impl Material<1>) {
        let node_count = self.node_count();
        if let Gauss::Edge(gauss_edge) = gauss {
            for row in gauss_edge.gauss_matrix().row_iter() {
                let xi = row[1];
                let w = row[0];
                let gauss_result =
                    gauss_edge.linear_shape_func_calc(self.nodes_coordinates(), [xi]);
                let JxW = gauss_result.det_j * w;
                for i in 0..node_count {
                    for j in 0..node_count {
                        // 这里要对高斯积分进行累加
                        self.K_mut()[(i, j)] += gauss_result.shp_grad[j]
                            * gauss_result.shp_grad[i]
                            * mat.get_constitutive_matrix()[(0, 0)]
                            * JxW;
                    }
                }
            }
        } else {
            panic!("gauss input not match this element")
        }
    }
}

impl StructureElement<1, 3> for Edge3 {
    fn structure_stiffness_calc(&mut self, gauss: &Gauss, mat: &impl Material<1>) {
        let node_count = self.node_count();
        if let Gauss::Edge(gauss_edge) = gauss {
            for row in gauss_edge.gauss_matrix().row_iter() {
                let xi = row[1];
                let w = row[0];
                let gauss_result =
                    gauss_edge.square_shape_func_calc(self.nodes_coordinates(), [xi]);
                let JxW = gauss_result.det_j * w;
                for i in 0..node_count {
                    for j in 0..node_count {
                        // 这里要对高斯积分进行累加
                        self.K_mut()[(i, j)] += gauss_result.shp_grad[j]
                            * gauss_result.shp_grad[i]
                            * mat.get_constitutive_matrix()[(0, 0)]
                            * JxW;
                    }
                }
            }
        } else {
            panic!("gauss input not match this element")
        }
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange1DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, SMatrix};

    use crate::{
        base::{elements::GeneralElement, gauss::GaussEdge},
        materials::IsotropicLinearElastic1D,
    };

    use super::*;

    #[test]
    fn structure_edge2_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 5, "edge2");
        let mut edge2 = Edge2::new(1);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic1D::new(1.0e9, 1.0);
        let gauss = Gauss::Edge(GaussEdge::new(2));
        for element_number in 0..mesh.element_count() {
            edge2.update(element_number, mesh.elements(), mesh.nodes());
            edge2.structure_stiffness_calc(&gauss, &mat);
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
        let mut edge3 = Edge3::new(1);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mat = IsotropicLinearElastic1D::new(1.0, 1.0);
        let gauss = Gauss::Edge(GaussEdge::new(2));
        for element_number in 0..mesh.element_count() {
            edge3.update(element_number, mesh.elements(), mesh.nodes());
            edge3.structure_stiffness_calc(&gauss, &mat);
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
