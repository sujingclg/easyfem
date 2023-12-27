use crate::base::{
    gauss::{Gauss, GaussResult},
    primitives::{Edge, PrimitiveBase},
};

use super::PoissonElement;

impl<const N: usize> PoissonElement<N, 1> for Edge<N> {
    fn poisson_stiffness_calc(&mut self, gauss: &impl Gauss<N, 1>, f: f64) {
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
                    self.K_mut()[(i, j)] += shp_grad[j] * shp_grad[i] * JxW;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange1DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, SMatrix};

    use crate::{
        base::{
            gauss::GaussEdge2,
            primitives::{Edge2, GeneralElement},
        },
        poisson::elements::PoissonElement,
    };

    #[test]
    fn poisson_edge2_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 10, "edge2");
        let mut edge2 = Edge2::new(1);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        // let gauss = &Gauss::Edge(GaussEdge::new(2));
        let gauss = GaussEdge2::new(2);

        for element_number in 0..mesh.element_count() {
            edge2.update(element_number, mesh.elements(), mesh.nodes());
            edge2.poisson_stiffness_calc(&gauss, 200.0);
            edge2.assemble(&mut stiffness_matrix, &mut right_vector);
        }

        // boundary conditions
        stiffness_matrix[(0, 0)] += 1e16;
        right_vector[0] += 0.5 * 1e16;
        stiffness_matrix[(n_dofs - 1, n_dofs - 1)] += 1e16;
        right_vector[n_dofs - 1] += 5.0 * 1e16;

        let answer = SMatrix::<f64, 11, 1>::from_column_slice(&[
            0.50, 9.95, 17.40, 22.85, 26.30, 27.75, 27.20, 24.65, 20.10, 13.55, 5.00,
        ]);
        let err = 1e-13;
        if let Some(phi) = stiffness_matrix.lu().solve(&right_vector) {
            println!("{:.2}", &phi);
            assert!(phi.relative_eq(&answer, err, err));
        }
    }
}
