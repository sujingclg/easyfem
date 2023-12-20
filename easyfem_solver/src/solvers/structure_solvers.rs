use easyfem_mesh::Mesh;
use nalgebra::{DMatrix, DVector, MatrixXx3};

use crate::{
    elements::{GeneralElement, StructureElement},
    materials::Material,
};

/// 边界条件, 只能是力边界条件或位移边界条件(二选一), T为节点自由度数
pub struct StructureBoundaryCondition<const T: usize> {
    node_id: usize,
    boundary_condition: [f64; T],
    is_force: bool, // 是否是力边界条件
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
        element_node_matrix: &DMatrix<usize>,    // 单元节点矩阵
        node_coordinate_matrix: &MatrixXx3<f64>, // 节点坐标矩阵
        element: &mut (impl GeneralElement + StructureElement<3>),
        // materialsMap: &HashMap<usize, Box<dyn Material<3>>>,
        mat: &impl Material<3>,
    ) {
        for element_number in 0..element_node_matrix.nrows() {
            element.update(element_number, element_node_matrix, node_coordinate_matrix);
            element.structure_calculate(mat);
            element.assemble(&mut self.stiffness_matrix);
        }
    }

    pub fn apply_boundary_conditions(
        &mut self,
        boundary_conditions: &Vec<StructureBoundaryCondition<2>>,
    ) {
        let penalty = 1.0e20;
        boundary_conditions.iter().for_each(|bc| {
            let x = bc.node_id * 2;
            let y = bc.node_id * 2 + 1;
            if bc.is_force {
                self.force_vector[x] = bc.boundary_condition[0];
                self.force_vector[y] = bc.boundary_condition[1];
            } else {
                let k_xx = self.stiffness_matrix[(x, x)];
                let k_yy = self.stiffness_matrix[(y, y)];
                self.stiffness_matrix[(x, x)] = k_xx * penalty;
                self.stiffness_matrix[(y, y)] = k_yy * penalty;
                self.force_vector[x] = penalty * k_xx * bc.boundary_condition[0];
                self.force_vector[y] = penalty * k_yy * bc.boundary_condition[1];
            }
        });
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

    pub fn display_displacement_vector(&self) {
        if let Some(d) = &self.displacement_vector {
            println!("{:.3e}", d);
        }
    }
}

struct Structure3DSolver<'a> {
    mesh: &'a dyn Mesh,
}

impl<'a> Structure3DSolver<'a> {
    pub fn new(mesh: &'a dyn Mesh) -> Self {
        Structure3DSolver { mesh }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, MatrixXx3};

    use crate::{
        elements::Quad4,
        materials::{IsotropicLinearElastic2D, PlaneCondition},
    };

    use super::{Structure2DSolver, StructureBoundaryCondition};

    #[test]
    fn structure_2d_solver_test_1() {
        let F = 1.0e5;
        let bcs = vec![
            StructureBoundaryCondition {
                node_id: 0,
                boundary_condition: [0.0, -F / 2.0],
                is_force: true,
            },
            StructureBoundaryCondition {
                node_id: 1,
                boundary_condition: [0.0, -F / 2.0],
                is_force: true,
            },
            StructureBoundaryCondition {
                node_id: 4,
                boundary_condition: [0.0, 0.0],
                is_force: false,
            },
            StructureBoundaryCondition {
                node_id: 5,
                boundary_condition: [0.0, 0.0],
                is_force: false,
            },
        ];

        let mut solver = Structure2DSolver::new(12);

        let element_node_matrix = DMatrix::from_row_slice(
            2,
            4,
            &[
                2, 4, 5, 3, // 0
                0, 2, 3, 1, // 1
            ],
        );
        let node_coordinate_matrix = MatrixXx3::from_row_slice(&[
            2.0, 1.0, 0.0, // 0
            2.0, 0.0, 0.0, // 1
            1.0, 1.0, 0.0, // 2
            1.0, 0.0, 0.0, // 3
            0.0, 1.0, 0.0, // 4
            0.0, 0.0, 0.0, // 5
        ]);
        let mut quad4 = Quad4::new(2);
        let mat = IsotropicLinearElastic2D::new(1.0e7, 1.0 / 3.0, PlaneCondition::PlaneStress, 0.1);
        solver.stiffness_calculate(
            &element_node_matrix,
            &node_coordinate_matrix,
            &mut quad4,
            &mat,
        );
        solver.display_stiffness_matrix();
        solver.apply_boundary_conditions(&bcs);
        solver.solve();
        solver.display_displacement_vector();
    }
}
