pub mod edge;

use easyfem_mesh::Mesh;
use nalgebra::{DMatrix, DVector};

trait DiffusionBase {
    fn stiffness_calculate(
        &mut self,
        mesh: &impl Mesh,
        stiffness_matrix: &mut DMatrix<f64>,
        right_vector: &mut DVector<f64>,
        prev_solution: &DVector<f64>,
    );

    fn boundary_calculate(&self, right_vector: &mut DVector<f64>);
}

trait DiffusionSolver: DiffusionBase {
    fn iteratively_solve(&mut self, mesh: &impl Mesh, total_step: usize) {
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);
        let mut prev_solution = DVector::from_element(n_dofs, 1e-3);
        let mut total_solutions = DMatrix::zeros(n_dofs, total_step);

        for step in 0..total_step {
            stiffness_matrix.fill(0.0);
            right_vector.fill(0.0);
            self.stiffness_calculate(
                mesh,
                &mut stiffness_matrix,
                &mut right_vector,
                &prev_solution,
            );
            self.boundary_calculate(&mut right_vector);
            if !stiffness_matrix.solve_lower_triangular_mut(&mut right_vector) {
                panic!("fail to solve matrix");
            }
            prev_solution.set_column(0, &right_vector);
            total_solutions.set_column(step, &right_vector);
        }
        println!("{:.2e}", total_solutions);
    }
}

#[cfg(test)]
mod tests {}
