// https://github.com/ahojukka5/gmshparser

use core::fmt;
use std::{fs::File, io};

use super::Mesh;

pub struct GmshReader {
    //
}

impl GmshReader {
    pub fn parse(mut lines_iter: io::Lines<io::BufReader<File>>) -> GmshMesh {
        let mut mesh = GmshMesh::new();
        while let Some(Ok(line)) = lines_iter.next() {
            if line == "$MeshFormat" {
                Self::parse_mesh_format(&mut mesh, &mut lines_iter);
            }
            if line == "$Nodes" {
                // Self::parse_nodes(&mut mesh, &mut lines_iter);
            }
            if line == "$Elements" {
                // Self::parse_elements(&mut mesh, &mut lines_iter);
            }
        }
        mesh
    }

    fn parse_mesh_format(mesh: &mut GmshMesh, lines_iter: &mut io::Lines<io::BufReader<File>>) {
        if let Some(Ok(line)) = lines_iter.next() {
            let s: Vec<&str> = line.trim().split_whitespace().collect();
            println!("{:?}", s);
            println!("{:?}", mesh.get_nodes());
            // mesh.set_version(float(s[0]));
            // mesh.set_ascii(int(s[1]) == 0);
            // mesh.set_precision(int(s[2]));
        }
    }

    // fn parse_nodes(mesh: &mut GmshMesh, lines_iter: &mut io::Lines<io::BufReader<File>>) {
    //     //
    // }

    // fn parse_elements(mesh: &mut GmshMesh, lines_iter: &mut io::Lines<io::BufReader<File>>) {
    //     //
    // }
}

#[derive(Default)]
pub struct GmshMesh {
    // node_count: i32,
    // min_node_tag: i32,
    // max_node_tag: i32,
    // element_count: i32,
    // min_element_tag: i32,
    // max_element_tag: i32,
}

impl GmshMesh {
    pub fn new() -> Self {
        GmshMesh::default()
    }
}

impl Mesh for GmshMesh {
    fn get_nodes(&self) {
        //
    }

    fn get_elements(&self) {
        //
    }
}

impl fmt::Display for GmshMesh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Display")
    }
}

#[cfg(test)]
mod tests {
    use crate::base::utils::read_lines;

    use super::GmshReader;

    #[test]
    fn test_gmash_reader() {
        // let contents =
        //     fs::read_to_string("rect.msh").expect("Should have been able to read the file");
        // let reader = GmshReader::new();
        // reader.parse(contents);

        if let Ok(lines) = read_lines("testmesh.msh") {
            let _mesh = GmshReader::parse(lines);
        }
    }
}
