use std::{collections::HashMap, fmt};

use nalgebra::{DMatrix, MatrixXx3, SMatrix};

use crate::Mesh;

pub struct Lagrange3DMesh {
    connectivity_matrix: DMatrix<usize>,            // 单元节点矩阵
    node_coordinate_matrix: MatrixXx3<f64>,         // 节点坐标矩阵
    boundary_node_ids: HashMap<String, Vec<usize>>, // 2d网格边界上的点
}

impl Lagrange3DMesh {
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        zmin: f64,
        zmax: f64,
        nz: usize,
        mesh_type: &str,
    ) -> Self {
        match mesh_type {
            "cube8" => Self::get_cube8_mesh(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz),
            &_ => todo!(),
        }
    }

    fn get_cube8_mesh(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        zmin: f64,
        zmax: f64,
        nz: usize,
    ) -> Self {
        let order = 1;
        let dx = (xmax - xmin) / (nx * order) as f64;
        let dy = (ymax - ymin) / (ny * order) as f64;
        let dz = (zmax - zmin) / (nz * order) as f64;
        let mut boundary_node_ids = HashMap::new();
        let mut node_coordinate_matrix = MatrixXx3::zeros((nx + 1) * (ny + 1) * (nz + 1));
        for k in 0..nz + 1 {
            for j in 0..ny + 1 {
                for i in 0..nx + 1 {
                    let node_id = k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
                    node_coordinate_matrix[(node_id, 0)] = xmin + dx * i as f64;
                    node_coordinate_matrix[(node_id, 1)] = ymin + dy * j as f64;
                    node_coordinate_matrix[(node_id, 2)] = zmin + dz * k as f64;
                    if i == 0 {
                        boundary_node_ids
                            .entry(String::from("back"))
                            .or_insert(Vec::new())
                            .push(node_id);
                    }
                    if i == nx {
                        boundary_node_ids
                            .entry(String::from("front"))
                            .or_insert(Vec::new())
                            .push(node_id);
                    }
                    if j == 0 {
                        boundary_node_ids
                            .entry(String::from("left"))
                            .or_insert(Vec::new())
                            .push(node_id);
                    }
                    if j == ny {
                        boundary_node_ids
                            .entry(String::from("right"))
                            .or_insert(Vec::new())
                            .push(node_id);
                    }
                    if k == 0 {
                        boundary_node_ids
                            .entry(String::from("bottom"))
                            .or_insert(Vec::new())
                            .push(node_id);
                    }
                    if k == nz {
                        boundary_node_ids
                            .entry(String::from("top"))
                            .or_insert(Vec::new())
                            .push(node_id);
                    }
                }
            }
        }

        let mut connectivity_matrix = DMatrix::zeros(nx * ny * nz, 8);

        for k in 1..nz + 1 {
            for j in 1..ny + 1 {
                for i in 1..nx + 1 {
                    let e = (k - 1) * ny * nx + (j - 1) * nx + i - 1;
                    let e0 = (k - 1) * (ny + 1) * (nx + 1) + (j - 1) * (nx + 1) + i - 1;
                    let e1 = e0 + 1;
                    let e2 = e1 + nx + 1;
                    let e3 = e2 - 1;
                    let e4 = e0 + (nx + 1) * (ny + 1);
                    let e5 = e4 + 1;
                    let e6 = e5 + nx + 1;
                    let e7 = e6 - 1;
                    connectivity_matrix.set_row(
                        e,
                        &SMatrix::<usize, 1, 8>::from_row_slice(&[e0, e1, e2, e3, e4, e5, e6, e7]),
                    );
                }
            }
        }

        Lagrange3DMesh {
            connectivity_matrix,
            node_coordinate_matrix,
            boundary_node_ids,
        }
    }

    pub fn get_boundary_node_ids(&self) -> &HashMap<String, Vec<usize>> {
        &self.boundary_node_ids
    }
}

impl Mesh for Lagrange3DMesh {
    fn get_elements(&self) -> &DMatrix<usize> {
        &self.connectivity_matrix
    }

    fn get_nodes(&self) -> &MatrixXx3<f64> {
        &self.node_coordinate_matrix
    }

    fn get_element_count(&self) -> usize {
        self.connectivity_matrix.nrows()
    }

    fn get_node_count(&self) -> usize {
        self.node_coordinate_matrix.nrows()
    }
}

impl fmt::Display for Lagrange3DMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.2}\n{}\n{:?}",
            self.node_coordinate_matrix, self.connectivity_matrix, self.boundary_node_ids
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube8_test_1() {
        let mesh = Lagrange3DMesh::new(0.0, 1.0, 2, 0.0, 1.0, 2, 0.0, 1.0, 2, "cube8");
        println!("{}", mesh);
    }
}
