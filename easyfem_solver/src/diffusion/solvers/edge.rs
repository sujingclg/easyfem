use std::collections::HashMap;

use easyfem_mesh::Mesh;
use nalgebra::{DMatrix, DVector};

use crate::diffusion::elements::{DiffusionEdge2, DiffusionEdge3, DiffusionElement};

use super::{DiffusionBase, DiffusionSolver};

pub struct DiffusionEdgeSolver {
    diffusivity: f64,
    dt: f64,
    j0: f64,
    edges: HashMap<String, Box<dyn DiffusionElement>>,
    // total_solutions: Option<DMatrix<f64>>,
}

impl DiffusionEdgeSolver {
    pub fn new(diffusivity: f64, dt: f64, j0: f64) -> Self {
        let mut edges: HashMap<String, Box<dyn DiffusionElement>> = HashMap::new();
        edges.insert(String::from("edge2"), Box::new(DiffusionEdge2::new(1, 2)));
        edges.insert(String::from("edge3"), Box::new(DiffusionEdge3::new(1, 2)));
        edges.insert(String::from("edge4"), Box::new(DiffusionEdge3::new(1, 2)));

        DiffusionEdgeSolver {
            diffusivity,
            dt,
            j0,
            edges,
            // total_solutions: None,
        }
    }

    // pub fn display_solutions(&self) {
    //     if let Some(solutions) = &self.total_solutions {
    //         println!("{:.2e}", solutions);
    //     }
    // }
}

impl DiffusionBase for DiffusionEdgeSolver {
    fn stiffness_calculate(
        &mut self,
        mesh: &impl Mesh,
        stiffness_matrix: &mut DMatrix<f64>,
        right_vector: &mut DVector<f64>,
        prev_solution: &DVector<f64>,
    ) {
        if let Some(element) = self.edges.get_mut("edge2") {
            for element_number in 0..mesh.element_count() {
                element.update(element_number, mesh.elements(), mesh.nodes());
                element.diffusion_stiffness_calc(self.diffusivity, self.dt, prev_solution);
                element.assemble(stiffness_matrix, right_vector);
            }
        }
    }

    fn boundary_calculate(&self, right_vector: &mut DVector<f64>) {
        let n_dofs = right_vector.nrows();
        right_vector[n_dofs - 1] += self.j0;
    }
}

impl DiffusionSolver for DiffusionEdgeSolver {}

#[cfg(test)]
mod tests {
    use easyfem_mesh::Lagrange1DMesh;

    use super::*;

    #[test]
    fn solver_test() {
        let mut solver = DiffusionEdgeSolver::new(0.5, 1e-3, 0.005);
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 10, "edge2");
        solver.iteratively_solve(&mesh, 20);
        // solver.display_solutions();
    }
}
