use crate::base::{
    gauss::{Gauss, GaussResult},
    primitives::{PrimitiveBase, Quad},
};

use super::PoissonElement;

impl<const N: usize> PoissonElement<N, 2> for Quad<N> {
    fn poisson_stiffness_calc(&mut self, gauss: &impl Gauss<N, 2>, f: f64) {
        // let node_dof = self.node_dof();
        let node_count = self.node_count();
        for (w, gauss_point) in gauss.gauss_vector() {
            let GaussResult {
                shp_val,
                shp_grad,
                det_j,
            } = gauss.shape_func_calc(self.nodes_coordinates(), gauss_point);
            let JxW = det_j * w;
            // TODO: 目前此方程没有考虑节点自由度
            for i in 0..node_count {
                self.F_mut()[i] += f * shp_val[i] * JxW;
                for j in 0..node_count {
                    self.K_mut()[(i, j)] += (shp_grad[(j, 0)] * shp_grad[(i, 0)]
                        + shp_grad[(j, 1)] * shp_grad[(i, 1)])
                        * JxW;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange2DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, SMatrix};

    use crate::{
        base::{
            gauss::GaussQuad4,
            primitives::{GeneralElement, Quad4},
        },
        poisson::elements::PoissonElement,
    };

    #[test]
    fn poisson_quad4_test() {
        let mesh = Lagrange2DMesh::new(0.0, 1.0, 5, 0.0, 1.0, 5, "quad4");
        let mut quad4 = Quad4::new(1);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let gauss = GaussQuad4::new(2);

        for element_number in 0..mesh.element_count() {
            quad4.update(element_number, mesh.elements(), mesh.nodes());
            quad4.poisson_stiffness_calc(&gauss, 200.0);
            quad4.assemble(&mut stiffness_matrix, &mut right_vector);
        }

        // boundary conditions
        let penalty = 1.0e16;
        if let Some(leftnodes) = mesh.get_boundary_node_ids().get("left") {
            for id in leftnodes {
                stiffness_matrix[(*id, *id)] += penalty;
                right_vector[*id] += 5.0 * penalty;
            }
        }
        if let Some(rightnodes) = mesh.get_boundary_node_ids().get("right") {
            for id in rightnodes {
                stiffness_matrix[(*id, *id)] += penalty;
                right_vector[*id] += 5.0 * penalty;
            }
        }
        if let Some(rightnodes) = mesh.get_boundary_node_ids().get("top") {
            for id in rightnodes {
                stiffness_matrix[(*id, *id)] += penalty;
                right_vector[*id] += 5.0 * penalty;
            }
        }
        if let Some(rightnodes) = mesh.get_boundary_node_ids().get("bottom") {
            for id in rightnodes {
                stiffness_matrix[(*id, *id)] += penalty;
                right_vector[*id] += 5.0 * penalty;
            }
        }

        let answer = SMatrix::<f64, 36, 1>::from_column_slice(&[
            5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000, 12.263, 14.947, 14.947, 12.263, 5.000,
            5.000, 14.947, 19.211, 19.211, 14.947, 5.000, 5.000, 14.947, 19.211, 19.211, 14.947,
            5.000, 5.000, 12.263, 14.947, 14.947, 12.263, 5.000, 5.000, 5.000, 5.000, 5.000, 5.000,
            5.000,
        ]);
        let err = 1e-4;
        if let Some(phi) = stiffness_matrix.lu().solve(&right_vector) {
            println!(
                "{:.3}",
                SMatrix::<f64, 6, 6>::from_column_slice(&phi.as_slice())
            );
            assert!(phi.relative_eq(&answer, err, err));
        }
    }
}
