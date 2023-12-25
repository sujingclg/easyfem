use nalgebra::{DMatrix, MatrixXx3};

mod lagrange_1d_mesh;
pub use lagrange_1d_mesh::Lagrange1DMesh;

mod lagrange_2d_mesh;
pub use lagrange_2d_mesh::Lagrange2DMesh;

mod lagrange_3d_mesh;
pub use lagrange_3d_mesh::Lagrange3DMesh;

pub trait Mesh {
    fn elements(&self) -> &DMatrix<usize>;

    fn nodes(&self) -> &MatrixXx3<f64>;

    fn element_count(&self) -> usize;

    fn node_count(&self) -> usize;
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
