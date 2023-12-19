mod gmsh;

pub use gmsh::{GmshMesh, GmshReader};

pub trait Mesh {
    fn get_nodes(&self);
    fn get_elements(&self);
}
