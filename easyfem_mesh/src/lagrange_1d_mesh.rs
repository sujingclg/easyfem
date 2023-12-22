use std::{fmt, usize};

use nalgebra::{DMatrix, MatrixXx3};

use crate::Mesh;

pub struct Lagrange1DMesh {
    element_connectivity_matrix: DMatrix<usize>, // 单元节点矩阵
    node_coordinate_matrix: MatrixXx3<f64>,      // 节点坐标矩阵
}

impl Lagrange1DMesh {
    pub fn new(xmin: f64, xmax: f64, nx: usize, mesh_type: &str) -> Self {
        if nx <= 0 {
            panic!("language 1d mesh error");
        }

        let (order, element_connectivity_matrix) = match mesh_type {
            "edge2" => (1, DMatrix::from_fn(nx, 2, |r, c| r + c)),
            "edge3" => (
                2,
                DMatrix::from_fn(nx, 3, |r, c| {
                    if c == 0 {
                        r * 2 + c
                    } else if c == 1 {
                        r * 2 + c + 1
                    } else {
                        r * 2 + c - 1
                    }
                }),
            ),
            _ => todo!(),
        };
        let dx = (xmax - xmin) / (nx * order) as f64;
        let node_coordinate_matrix = MatrixXx3::from_fn(nx * order + 1, |r, c| {
            let result = if c == 0 { dx * r as f64 } else { 0.0 };
            result
        });
        Lagrange1DMesh {
            element_connectivity_matrix,
            node_coordinate_matrix,
        }
    }
}

impl Mesh for Lagrange1DMesh {
    fn get_elements(&self) -> &DMatrix<usize> {
        &self.element_connectivity_matrix
    }

    fn get_nodes(&self) -> &MatrixXx3<f64> {
        &self.node_coordinate_matrix
    }

    fn get_element_count(&self) -> usize {
        self.element_connectivity_matrix.nrows()
    }
}

impl fmt::Display for Lagrange1DMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.2}\n{}",
            self.node_coordinate_matrix, self.element_connectivity_matrix,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge2_test_1() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 5, "edge2");
        println!("{}", mesh);
    }

    #[test]
    fn edge3_test_2() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 2, "edge3");
        println!("{}", mesh);
    }
}
