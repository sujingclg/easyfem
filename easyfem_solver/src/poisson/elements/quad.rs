use crate::base::{
    gauss::{Gauss, GaussQuad4, GaussResult},
    primitives::{GeneralElement, PrimitiveBase, Quad},
};

use super::PoissonElement;

pub struct PoissonQuad<const N: usize> {
    quad: Quad<N>,
    gauss: Box<dyn Gauss<N, 2>>,
}

pub type PoissonQuad4 = PoissonQuad<4>;
impl PoissonQuad4 {
    pub fn new(node_dof: usize, gauss_deg: usize) -> Self {
        PoissonQuad4 {
            quad: Quad::<4>::new(node_dof),
            gauss: Box::new(GaussQuad4::new(gauss_deg)),
        }
    }
}

impl<const N: usize> PoissonElement for PoissonQuad<N> {
    fn update(
        &mut self,
        element_number: usize, // 单元编号, 即单元的全局索引
        connectivity_matrix: &nalgebra::DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &nalgebra::MatrixXx3<f64>, // 全局节点-坐标矩阵
    ) {
        self.quad
            .update(element_number, connectivity_matrix, coordinate_matrix);
    }

    fn poisson_stiffness_calc(&mut self, f: f64) {
        // let node_dof = self.node_dof();
        let node_count = self.quad.node_count();
        for (w, gauss_point) in self.gauss.gauss_vector() {
            let GaussResult {
                shp_val,
                shp_grad,
                det_j,
            } = self
                .gauss
                .shape_func_calc(self.quad.nodes_coordinates(), gauss_point);
            let JxW = det_j * w;
            // TODO: 目前此方程没有考虑节点自由度
            for i in 0..node_count {
                self.quad.F_mut()[i] += f * shp_val[i] * JxW;
                for j in 0..node_count {
                    self.quad.K_mut()[(i, j)] += (shp_grad[(j, 0)] * shp_grad[(i, 0)]
                        + shp_grad[(j, 1)] * shp_grad[(i, 1)])
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
        self.quad.assemble(stiffness_matrix, right_vector);
    }
}

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange2DMesh, Mesh};
    use nalgebra::{DMatrix, DVector, SMatrix};

    use super::*;

    #[test]
    fn poisson_quad4_test() {
        let mesh = Lagrange2DMesh::new(0.0, 1.0, 5, 0.0, 1.0, 5, "quad4");
        let mut quad4 = PoissonQuad4::new(1, 2);
        let n_dofs = mesh.node_count();
        let mut stiffness_matrix = DMatrix::zeros(n_dofs, n_dofs);
        let mut right_vector = DVector::zeros(n_dofs);

        for element_number in 0..mesh.element_count() {
            quad4.update(element_number, mesh.elements(), mesh.nodes());
            quad4.poisson_stiffness_calc(200.0);
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
