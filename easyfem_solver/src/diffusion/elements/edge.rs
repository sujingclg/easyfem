use nalgebra::DVector;

use crate::base::{
    elements::{Edge, ElementBase, GeneralElement},
    gauss::{Gauss, GaussResult},
};

use super::DiffusionElement;

impl<const N: usize> DiffusionElement<N, 1> for Edge<N> {
    fn diffusion_stiffness_calc(
        &mut self,
        gauss: &impl Gauss<N, 1>,
        diffusivity: f64,
        dt: f64,
        prev_solution: &DVector<f64>,
    ) {
        let node_count = self.node_count();
        for (w, gauss_point) in gauss.gauss_vector() {
            let GaussResult {
                shp_val,
                shp_grad,
                det_j,
            } = gauss.shape_func_calc(self.nodes_coordinates(), gauss_point);
            let JxW = det_j * w;
            for i in 0..node_count {
                let prev_f = prev_solution[self.global_node_id(i)];
                self.F_mut()[i] += prev_f / dt * shp_val[i] * JxW;
                for j in 0..node_count {
                    self.K_mut()[(i, j)] += (1.0 / dt) * shp_val[j] * shp_val[i] * JxW
                        + diffusivity * shp_grad[j] * shp_grad[i] * JxW;
                }
            }
        }
    }
}

// impl DiffusionElement<3, 1> for Edge3 {
//     fn diffusion_stiffness_calc(
//         &mut self,
//         gauss: &Gauss,
//         diffusivity: f64,
//         dt: f64,
//         prev_solution: &DVector<f64>,
//     ) {
//         let node_count = self.node_count();
//         if let Gauss::Edge(gauss_edge) = gauss {
//             for row in gauss_edge.gauss_matrix().row_iter() {
//                 let xi = row[1];
//                 let w = row[0];
//                 let GaussResult {
//                     shp_val,
//                     shp_grad,
//                     det_j,
//                 } = gauss_edge.square_shape_func_calc(self.nodes_coordinates(), [xi]);
//                 let JxW = det_j * w;
//                 for i in 0..node_count {
//                     let prev_f = prev_solution[self.global_node_id(i)];
//                     self.F_mut()[i] += prev_f / dt * shp_val[i] * JxW;
//                     for j in 0..node_count {
//                         self.K_mut()[(i, j)] += (1.0 / dt * shp_val[j] * shp_val[i]
//                             + diffusivity * shp_grad[j] * shp_grad[i])
//                             * JxW;
//                     }
//                 }
//             }
//         } else {
//             panic!("gauss input not match this element")
//         }
//     }
// }

// impl DiffusionElement<4, 1> for Edge4 {
//     fn diffusion_stiffness_calc(
//         &mut self,
//         gauss: &Gauss,
//         diffusivity: f64,
//         dt: f64,
//         prev_solution: &DVector<f64>,
//     ) {
//         let node_count = self.node_count();
//         if let Gauss::Edge(gauss_edge) = gauss {
//             for row in gauss_edge.gauss_matrix().row_iter() {
//                 let xi = row[1];
//                 let w = row[0];
//                 let GaussResult {
//                     shp_val,
//                     shp_grad,
//                     det_j,
//                 } = gauss_edge.cubic_shape_func_calc(self.nodes_coordinates(), [xi]);
//                 let JxW = det_j * w;
//                 for i in 0..node_count {
//                     let prev_f = prev_solution[self.global_node_id(i)];
//                     self.F_mut()[i] += prev_f / dt * shp_val[i] * JxW;
//                     for j in 0..node_count {
//                         self.K_mut()[(i, j)] += (1.0 / dt * shp_val[j] * shp_val[i]
//                             + diffusivity * shp_grad[j] * shp_grad[i])
//                             * JxW;
//                     }
//                 }
//             }
//         } else {
//             panic!("gauss input not match this element")
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange1DMesh, Mesh};
    use nalgebra::{DMatrix, DVector};

    use crate::{
        base::{
            elements::{Edge2, Edge3, Edge4, GeneralElement},
            gauss::{GaussEdge2, GaussEdge3, GaussEdge4},
        },
        diffusion::elements::DiffusionElement,
    };

    #[test]
    fn diffusion_edge2_test() {
        let D = 0.5; // diffusion coefficient
        let dt = 1.0e-3;
        let total_step = 20;
        let j0 = 0.005;

        let mesh = Lagrange1DMesh::new(0.0, 1.0, 10, "edge2");
        let mut edge2 = Edge2::new(1);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let gauss = GaussEdge2::new(3);

        let mut prev_solution = DVector::from_element(n_dofs, 1e-3);
        let mut total_solutions = DMatrix::zeros(n_dofs, total_step);

        for step in 0..total_step {
            stiffness_matrix.fill(0.0);
            right_vector.fill(0.0);
            for element_number in 0..mesh.element_count() {
                edge2.update(element_number, mesh.elements(), mesh.nodes());
                edge2.diffusion_stiffness_calc(&gauss, D, dt, &prev_solution);
                edge2.assemble(&mut stiffness_matrix, &mut right_vector);
            }
            right_vector[n_dofs - 1] += j0;
            // println!("{:.3}", &right_vector);
            if !stiffness_matrix.solve_lower_triangular_mut(&mut right_vector) {
                panic!("fail to solve matrix");
            }
            prev_solution.set_column(0, &right_vector);
            // println!("{:.3}", &right_vector);
            total_solutions.set_column(step, &right_vector);
        }
        println!("{:.2e}", total_solutions);
    }

    #[test]
    fn diffusion_edge3_test() {
        let D = 0.5; // diffusion coefficient
        let dt = 1.0e-3;
        let total_step = 20;
        let j0 = 0.005;

        let mesh = Lagrange1DMesh::new(0.0, 1.0, 10, "edge3");
        let mut edge3 = Edge3::new(1);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let gauss = GaussEdge3::new(3);

        let mut prev_solution = DVector::from_element(n_dofs, 1e-3);
        let mut total_solutions = DMatrix::zeros(n_dofs, total_step);

        for step in 0..total_step {
            stiffness_matrix.fill(0.0);
            right_vector.fill(0.0);
            for element_number in 0..mesh.element_count() {
                edge3.update(element_number, mesh.elements(), mesh.nodes());
                edge3.diffusion_stiffness_calc(&gauss, D, dt, &prev_solution);
                edge3.assemble(&mut stiffness_matrix, &mut right_vector);
            }
            right_vector[n_dofs - 1] += j0;
            if !stiffness_matrix.solve_lower_triangular_mut(&mut right_vector) {
                panic!("fail to solve matrix");
            }
            prev_solution.set_column(0, &right_vector);
            total_solutions.set_column(step, &right_vector);
        }
        println!("{:.3e}", total_solutions);
    }

    #[test]
    fn diffusion_edge4_test() {
        let D = 0.5; // diffusion coefficient
        let dt = 1.0e-3;
        let total_step = 20;
        let j0 = 0.005;

        let mesh = Lagrange1DMesh::new(0.0, 1.0, 10, "edge4");
        let mut edge4 = Edge4::new(1);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let gauss = GaussEdge4::new(3);

        let mut prev_solution = DVector::from_element(n_dofs, 1e-3);
        let mut total_solutions = DMatrix::zeros(n_dofs, total_step);

        for step in 0..total_step {
            stiffness_matrix.fill(0.0);
            right_vector.fill(0.0);
            for element_number in 0..mesh.element_count() {
                edge4.update(element_number, mesh.elements(), mesh.nodes());
                edge4.diffusion_stiffness_calc(&gauss, D, dt, &prev_solution);
                edge4.assemble(&mut stiffness_matrix, &mut right_vector);
            }
            right_vector[n_dofs - 1] += j0;
            if !stiffness_matrix.solve_lower_triangular_mut(&mut right_vector) {
                panic!("fail to solve matrix");
            }
            prev_solution.set_column(0, &right_vector);
            total_solutions.set_column(step, &right_vector);
        }
        println!("{:.3e}", total_solutions);
    }
}
