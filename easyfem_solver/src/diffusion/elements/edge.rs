use nalgebra::DVector;

use crate::base::{
    gauss::{Gauss, GaussEdge2, GaussEdge3, GaussEdge4, GaussResult},
    primitives::{Edge, Primitive},
};

use super::DiffusionElement;

pub struct DiffusionEdge<const N: usize> {
    edge: Edge<N>,
    gauss: Box<dyn Gauss<N, 1>>,
}

pub type DiffusionEdge2 = DiffusionEdge<2>;
impl DiffusionEdge2 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        DiffusionEdge2 {
            edge: Edge::<2>::new(node_dof),
            gauss: Box::new(GaussEdge2::new(gauss_deg)),
        }
    }
}

pub type DiffusionEdge3 = DiffusionEdge<3>;
impl DiffusionEdge3 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        DiffusionEdge3 {
            edge: Edge::<3>::new(node_dof),
            gauss: Box::new(GaussEdge3::new(gauss_deg)),
        }
    }
}

pub type DiffusionEdge4 = DiffusionEdge<4>;
impl DiffusionEdge4 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        DiffusionEdge4 {
            edge: Edge::<4>::new(node_dof),
            gauss: Box::new(GaussEdge4::new(gauss_deg)),
        }
    }
}

impl<const N: usize> DiffusionElement for DiffusionEdge<N> {
    fn update(
        &mut self,
        element_number: usize, // 单元编号, 即单元的全局索引
        connectivity_matrix: &nalgebra::DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &nalgebra::MatrixXx3<f64>, // 全局节点-坐标矩阵
    ) {
        self.edge
            .update(element_number, connectivity_matrix, coordinate_matrix);
    }

    fn diffusion_stiffness_calc(
        &mut self,
        diffusivity: f64,
        dt: f64,
        prev_solution: &DVector<f64>,
    ) {
        let node_count = self.edge.node_count();
        for (w, gauss_point) in self.gauss.gauss_vector() {
            let GaussResult {
                shp_val,
                shp_grad,
                det_j,
            } = self
                .gauss
                .shape_func_calc(self.edge.nodes_coordinates(), gauss_point);
            let JxW = det_j * w;
            for i in 0..node_count {
                let prev_f = prev_solution[self.edge.global_node_id(i)];
                self.edge.F_mut()[i] += prev_f / dt * shp_val[i] * JxW;
                for j in 0..node_count {
                    self.edge.K_mut()[(i, j)] += (1.0 / dt) * shp_val[j] * shp_val[i] * JxW
                        + diffusivity * shp_grad[j] * shp_grad[i] * JxW;
                }
            }
        }
    }

    fn assemble(
        &mut self,
        stiffness_matrix: &mut nalgebra::DMatrix<f64>,
        right_vector: &mut DVector<f64>,
    ) {
        self.edge.assemble(stiffness_matrix, right_vector);
        self.edge.clean();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use easyfem_mesh::{Lagrange1DMesh, Mesh};
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn diffusion_edge2_test() {
        let D = 0.5; // diffusion coefficient
        let dt = 1.0e-3;
        let total_step = 20;
        let j0 = 0.005;

        let mesh = Lagrange1DMesh::new(0.0, 1.0, 20, "edge2");
        let mut edge2 = DiffusionEdge2::new(1, 2);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);

        let mut prev_solution = DVector::from_element(n_dofs, 1e-3);
        let mut total_solutions = DMatrix::zeros(n_dofs, total_step);

        for step in 0..total_step {
            stiffness_matrix.fill(0.0);
            right_vector.fill(0.0);
            for element_number in 0..mesh.element_count() {
                edge2.update(element_number, mesh.elements(), mesh.nodes());
                edge2.diffusion_stiffness_calc(D, dt, &prev_solution);
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
        let mut edge3 = DiffusionEdge3::new(1, 2);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);

        let mut prev_solution = DVector::from_element(n_dofs, 1e-3);
        let mut total_solutions = DMatrix::zeros(n_dofs, total_step);

        for step in 0..total_step {
            stiffness_matrix.fill(0.0);
            right_vector.fill(0.0);
            for element_number in 0..mesh.element_count() {
                edge3.update(element_number, mesh.elements(), mesh.nodes());
                edge3.diffusion_stiffness_calc(D, dt, &prev_solution);
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
        let mut edge4 = DiffusionEdge4::new(1, 3);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);

        let mut prev_solution = DVector::from_element(n_dofs, 1e-3);
        let mut total_solutions = DMatrix::zeros(n_dofs, total_step);

        for step in 0..total_step {
            stiffness_matrix.fill(0.0);
            right_vector.fill(0.0);
            for element_number in 0..mesh.element_count() {
                edge4.update(element_number, mesh.elements(), mesh.nodes());
                edge4.diffusion_stiffness_calc(D, dt, &prev_solution);
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
