use std::{collections::HashMap, fmt};

use nalgebra::{DMatrix, MatrixXx3, SMatrix};

use crate::Mesh;

pub struct Lagrange2DMesh {
    connectivity_matrix: DMatrix<usize>,            // 单元节点矩阵
    node_coordinate_matrix: MatrixXx3<f64>,         // 节点坐标矩阵
    boundary_node_ids: HashMap<String, Vec<usize>>, // 2d网格边界上的点
}

impl Lagrange2DMesh {
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        mesh_type: &str,
    ) -> Self {
        match mesh_type {
            "quad4" => Self::get_quad4_mesh(xmin, xmax, nx, ymin, ymax, ny),
            &_ => todo!(),
        }
    }

    fn get_quad4_mesh(xmin: f64, xmax: f64, nx: usize, ymin: f64, ymax: f64, ny: usize) -> Self {
        let order = 1;
        let dx = (xmax - xmin) / (nx * order) as f64;
        let dy = (ymax - ymin) / (ny * order) as f64;
        // let node_coordinate_matrix = MatrixXx3::from_fn((nx + 1) * (ny + 1), |r, c| {
        //     let result = if c == 0 {
        //         xmin + dx * (r % (nx + 1)) as f64
        //     } else if c == 1 {
        //         ymin + dy * (r / (nx + 1)) as f64
        //     } else {
        //         0.0
        //     };
        //     result
        // });
        let mut node_coordinate_matrix = MatrixXx3::zeros((nx + 1) * (ny + 1));
        let mut boundary_node_ids = HashMap::new();
        for j in 0..ny + 1 {
            for i in 0..nx + 1 {
                let node_id = j * (nx + 1) + i;
                node_coordinate_matrix[(node_id, 0)] = xmin + dx * i as f64;
                node_coordinate_matrix[(node_id, 1)] = ymin + dy * j as f64;
                if i == 0 {
                    boundary_node_ids
                        .entry(String::from("left"))
                        .or_insert(Vec::new())
                        .push(node_id);
                }
                if i == nx {
                    boundary_node_ids
                        .entry(String::from("right"))
                        .or_insert(Vec::new())
                        .push(node_id);
                }
                if j == 0 {
                    boundary_node_ids
                        .entry(String::from("bottom"))
                        .or_insert(Vec::new())
                        .push(node_id);
                }
                if j == ny {
                    boundary_node_ids
                        .entry(String::from("top"))
                        .or_insert(Vec::new())
                        .push(node_id);
                }
            }
        }
        let mut connectivity_matrix = DMatrix::zeros(nx * ny, 4);
        for j in 1..ny + 1 {
            for i in 1..nx + 1 {
                let e = (j - 1) * nx + i - 1;
                let e0 = (j - 1) * (nx + 1) + i - 1;
                let e1 = e0 + 1;
                let e2 = e1 + nx + 1;
                let e3 = e2 - 1;
                // 3 +-----+ 2
                //   |     |
                //   |     |
                // 0 +-----+ 1
                connectivity_matrix.set_row(
                    e,
                    &SMatrix::<usize, 1, 4>::from_row_slice(&[e0, e1, e2, e3]),
                );
            }
        }
        Lagrange2DMesh {
            connectivity_matrix,
            node_coordinate_matrix,
            boundary_node_ids,
        }
    }

    pub fn get_boundary_node_ids(&self) -> &HashMap<String, Vec<usize>> {
        &self.boundary_node_ids
    }
}

impl Mesh for Lagrange2DMesh {
    fn elements(&self) -> &DMatrix<usize> {
        &self.connectivity_matrix
    }

    fn nodes(&self) -> &MatrixXx3<f64> {
        &self.node_coordinate_matrix
    }

    fn element_count(&self) -> usize {
        self.connectivity_matrix.nrows()
    }

    fn node_count(&self) -> usize {
        self.node_coordinate_matrix.nrows()
    }
}

impl fmt::Display for Lagrange2DMesh {
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
    fn quad4_test_1() {
        let mesh = Lagrange2DMesh::new(0.0, 1.0, 5, 0.0, 1.0, 5, "quad4");
        println!("{}", mesh);
    }
}
