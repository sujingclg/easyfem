use nalgebra::{DMatrix, MatrixXx3};

mod lagrange_1d_mesh;
pub use lagrange_1d_mesh::Lagrange1DMesh;

mod lagrange_2d_mesh;
pub use lagrange_2d_mesh::Lagrange2DMesh;

mod lagrange_3d_mesh;
pub use lagrange_3d_mesh::Lagrange3DMesh;

pub trait Mesh {
    fn get_elements(&self) -> &DMatrix<usize>;

    fn get_nodes(&self) -> &MatrixXx3<f64>;

    fn get_element_count(&self) -> usize;

    fn get_node_count(&self) -> usize;
}

/// DOF -> degree 节点自由度
pub trait Mesh2<const DOF: usize> {
    fn get_nodes(&self);
    fn get_elements(&self);
}

// pub fn add(left: usize, right: usize) -> usize {
//     left + right
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
