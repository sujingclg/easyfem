mod lagrange_1d_mesh;
pub use lagrange_1d_mesh::Lagrange1DMesh;

pub trait Mesh {
    fn get_nodes(&self);
    fn get_elements(&self);
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
