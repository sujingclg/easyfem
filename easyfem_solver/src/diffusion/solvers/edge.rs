use easyfem_mesh::Mesh;
use nalgebra::{DMatrix, DVector};

use crate::{
    base::{
        gauss::GaussEdge2,
        primitives::{Edge2, GeneralElement},
    },
    diffusion::elements::DiffusionElement,
};

use super::{DiffusionBase, DiffusionSolver};

pub struct DiffusionEdgeSolver {
    diffusivity: f64,
    dt: f64,
    j0: f64,
    // total_solutions: Option<DMatrix<f64>>,
}

impl DiffusionEdgeSolver {
    pub fn new(diffusivity: f64, dt: f64, j0: f64) -> Self {
        DiffusionEdgeSolver {
            diffusivity,
            dt,
            j0,
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
        &self,
        mesh: &impl Mesh,
        stiffness_matrix: &mut DMatrix<f64>,
        right_vector: &mut DVector<f64>,
        prev_solution: &DVector<f64>,
    ) {
        // let gauss_edge = Gauss::Edge(GaussEdge::new(2));
        let gauss = GaussEdge2::new(2);
        let mut edge = Edge2::new(1);
        for element_number in 0..mesh.element_count() {
            edge.update(element_number, mesh.elements(), mesh.nodes());
            edge.diffusion_stiffness_calc(&gauss, self.diffusivity, self.dt, prev_solution);
            edge.assemble(stiffness_matrix, right_vector);
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
