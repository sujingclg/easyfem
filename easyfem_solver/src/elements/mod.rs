use nalgebra::{DMatrix, DVector, MatrixXx3};

use crate::base::gauss::Gauss;

pub trait Element {
    type CoordMatrix;

    fn gauss(&self) -> dyn Gauss<2, 1>;

    fn node_count(&self) -> usize;

    fn nodes_coordinates(&self) -> &Self::CoordMatrix;

    fn update(
        &mut self,
        element_number: usize,                // 单元编号, 即单元的全局索引
        connectivity_matrix: &DMatrix<usize>, // 全局单元-节点编号矩阵
        coordinate_matrix: &MatrixXx3<f64>,   // 全局节点-坐标矩阵
    );

    fn assemble(&mut self, stiffness_matrix: &mut DMatrix<f64>, right_vector: &mut DVector<f64>);

    fn K(&self) -> &DMatrix<f64>;
    fn K_mut(&mut self) -> &mut DMatrix<f64>;

    fn F(&self) -> &DVector<f64>;
    fn F_mut(&mut self) -> &mut DVector<f64>;
}
