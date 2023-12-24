// use easyfem_mesh::Mesh;
use nalgebra::{DMatrix, DVector, MatrixXx3};

use crate::{
    base_elements::{GeneralElement, StructureElement},
    materials::Material,
};

/// 边界条件, 力边界条件或位移边界条件, T为节点自由度数
pub enum StructureBoundaryCondition<const T: usize> {
    Force { node_id: usize, values: [f64; T] },
    Displacement { node_id: usize, values: [f64; T] },
}

pub trait StructureSolver {
    //
}

pub struct Structure2DSolver {
    stiffness_matrix: DMatrix<f64>,
    displacement_vector: Option<DVector<f64>>,
    force_vector: DVector<f64>,
}

impl Structure2DSolver {
    pub fn new(n_dofs: usize) -> Self {
        // let n_dofs = node_count * 2;
        Structure2DSolver {
            stiffness_matrix: DMatrix::zeros(n_dofs, n_dofs),
            displacement_vector: None,
            force_vector: DVector::zeros(n_dofs),
            // materialsMap: todo!(),
        }
    }

    /// N 为材料本构矩阵的阶数, 一维N=1, 二维N=3, 三维N=6
    pub fn stiffness_calculate(
        &mut self,
        connectivity_matrix: &DMatrix<usize>,    // 单元节点矩阵
        node_coordinate_matrix: &MatrixXx3<f64>, // 节点坐标矩阵
        element: &mut (impl GeneralElement + StructureElement<3>),
        // materialsMap: &HashMap<usize, Box<dyn Material<3>>>,
        mat: &impl Material<3>,
    ) {
        for element_number in 0..connectivity_matrix.nrows() {
            element.update(element_number, connectivity_matrix, node_coordinate_matrix);
            element.structure_stiffness_calculate(mat);
            element.assemble(&mut self.stiffness_matrix, &mut self.force_vector);
        }
    }

    pub fn apply_boundary_conditions(
        &mut self,
        boundary_conditions: &Vec<StructureBoundaryCondition<2>>,
    ) {
        use StructureBoundaryCondition::*;
        let penalty = 1.0e20;
        for bc in boundary_conditions {
            match bc {
                Force { node_id, values } => {
                    self.force_vector
                        .rows_mut(node_id * 2, 2)
                        .copy_from_slice(values);
                }
                Displacement { node_id, values } => {
                    let x = node_id * 2;
                    let y = node_id * 2 + 1;
                    let k_xx = self.stiffness_matrix[(x, x)];
                    let k_yy = self.stiffness_matrix[(y, y)];
                    self.stiffness_matrix[(x, x)] = k_xx * penalty;
                    self.stiffness_matrix[(y, y)] = k_yy * penalty;
                    self.force_vector[x] = penalty * k_xx * values[0];
                    self.force_vector[y] = penalty * k_yy * values[1];
                }
            }
        }
    }

    pub fn solve(&mut self) {
        // TODO: 使用 sprs 库中提供的 CG 求解器来解决稀疏对称对角矩阵的线性方程组
        let K = self.stiffness_matrix.clone();
        // if let Some(A) = K.cholesky() {
        //     self.displacement_vector = Some(A.solve(&self.load_vector));
        // }
        self.displacement_vector = K.lu().solve(&self.force_vector);
    }

    pub fn display_stiffness_matrix(&self) {
        println!("{:.3e}", self.stiffness_matrix);
    }
}

// struct Structure3DSolver<'a> {
//     mesh: &'a dyn Mesh,
// }

// impl<'a> Structure3DSolver<'a> {
//     pub fn new(mesh: &'a dyn Mesh) -> Self {
//         Structure3DSolver { mesh }
//     }
// }

#[cfg(test)]
mod tests {
    use easyfem_mesh::{Lagrange2DMesh, Mesh};
    use nalgebra::SMatrix;

    use crate::{
        base_elements::Quad4,
        gauss::GaussQuad,
        materials::{IsotropicLinearElastic2D, PlaneCondition::*},
    };

    use super::{Structure2DSolver, StructureBoundaryCondition::*};

    #[test]
    /// 曾攀 有限元分析基础教程 算例4.7.2
    fn structure_2d_test_1() {
        let F = 1.0e5;
        let mesh = Lagrange2DMesh::new(0.0, 2.0, 2, 0.0, 1.0, 1, "quad4");
        println!("{}", mesh);
        let mut bcs = vec![];
        if let Some(leftnodes) = mesh.get_boundary_node_ids().get("left") {
            leftnodes.iter().for_each(|id| {
                bcs.push(Displacement {
                    node_id: *id,
                    values: [0.0, 0.0],
                });
            })
        }
        if let Some(leftnodes) = mesh.get_boundary_node_ids().get("right") {
            leftnodes.iter().for_each(|id| {
                bcs.push(Force {
                    node_id: *id,
                    values: [0.0, -F / 2.0],
                });
            })
        }
        let mut solver = Structure2DSolver::new(12);
        let mut quad4 = Quad4::new(2, GaussQuad::new(2));
        let mat = IsotropicLinearElastic2D::new(1.0e7, 1.0 / 3.0, PlaneStress, 0.1);
        solver.stiffness_calculate(mesh.get_elements(), mesh.get_nodes(), &mut quad4, &mat);
        solver.apply_boundary_conditions(&bcs);
        solver.solve();
        let answer = SMatrix::<f64, 12, 1>::from_row_slice(&[
            -4.000e-21, -1.000e-21, -6.000e-1, -8.667e-1, -8.000e-1, -2.533e0, 4.000e-21,
            -1.000e-21, 6.000e-1, -8.667e-1, 8.000e-1, -2.533e0,
        ]);
        let err = 1e-3;
        if let Some(d) = solver.displacement_vector {
            println!("{:.3e}", &d);
            assert!(d.relative_eq(&answer, err, err));
        }
    }
}
