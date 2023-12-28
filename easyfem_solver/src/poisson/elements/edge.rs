use crate::base::{
    gauss::{Gauss, GaussEdge2, GaussEdge3, GaussResult},
    primitives::{Edge, Primitive},
};

use super::PoissonElement;

pub struct PoissonEdge<const N: usize> {
    edge: Edge<N>,
    gauss: Box<dyn Gauss<N, 1>>,
}

pub type PoissonEdge2 = PoissonEdge<2>;
impl PoissonEdge2 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        PoissonEdge2 {
            edge: Edge::<2>::new(node_dof),
            gauss: Box::new(GaussEdge2::new(gauss_deg)),
        }
    }
}

pub type PoissonEdge3 = PoissonEdge<3>;
impl PoissonEdge3 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        PoissonEdge3 {
            edge: Edge::<3>::new(node_dof),
            gauss: Box::new(GaussEdge3::new(gauss_deg)),
        }
    }
}

impl<const N: usize> PoissonElement for PoissonEdge<N> {
    fn update(
        &mut self,
        element_number: usize, // 单元编号, 即单元的全局索引
        connectivity_matrix: &nalgebra::DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &nalgebra::MatrixXx3<f64>, // 全局节点-坐标矩阵
    ) {
        self.edge
            .update(element_number, connectivity_matrix, coordinate_matrix);
    }

    fn poisson_stiffness_calc(&mut self, f: f64) {
        // let node_dof = self.node_dof();
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
            // TODO: 目前此方程没有考虑节点自由度
            for i in 0..node_count {
                self.edge.F_mut()[i] += f * shp_val[i] * JxW;
                for j in 0..node_count {
                    self.edge.K_mut()[(i, j)] += shp_grad[j] * shp_grad[i] * JxW;
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
        self.edge.clean();
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange1DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, SMatrix};

    use crate::poisson::elements::PoissonElement;

    use super::*;

    #[test]
    fn poisson_edge2_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 10, "edge2");
        let mut edge2 = PoissonEdge2::new(1, 2);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);

        for element_number in 0..mesh.element_count() {
            edge2.update(element_number, mesh.elements(), mesh.nodes());
            edge2.poisson_stiffness_calc(200.0);
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
