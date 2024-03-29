use std::{fmt, usize};

use nalgebra::{DMatrix, MatrixXx3};

use crate::Mesh;

pub struct Lagrange1DMesh {
    connectivity_matrix: DMatrix<usize>,    // 单元节点矩阵
    node_coordinate_matrix: MatrixXx3<f64>, // 节点坐标矩阵
}

impl Lagrange1DMesh {
    pub fn new(xmin: f64, xmax: f64, nx: usize, mesh_type: &str) -> Self {
        if nx <= 0 {
            panic!("language 1d mesh error");
        }

        let (order, connectivity_matrix) = match mesh_type {
            "edge2" => (1, DMatrix::from_fn(nx, 2, |r, c| r + c)),
            "edge3" => (
                2,
                DMatrix::from_fn(nx, 3, |r, c| {
                    if c == 0 {
                        r * 2
                    } else if c == 1 {
                        r * 2 + 2
                    } else {
                        r * 2 + c - 1
                    }
                }),
            ),
            "edge4" => (
                3,
                DMatrix::from_fn(nx, 4, |r, c| {
                    if c == 0 {
                        r * 3
                    } else if c == 1 {
                        r * 3 + 3
                    } else {
                        r * 3 + c - 1
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
            connectivity_matrix,
            node_coordinate_matrix,
        }
    }
}

impl Mesh for Lagrange1DMesh {
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

impl fmt::Display for Lagrange1DMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.2}\n{}",
            self.node_coordinate_matrix, self.connectivity_matrix,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge2_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 5, "edge2");
        println!("{}", mesh);
    }

    #[test]
    fn edge3_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.0, 2, "edge3");
        println!("{}", mesh);
    }

    #[test]
    fn edge4_test() {
        let mesh = Lagrange1DMesh::new(0.0, 1.8, 3, "edge4");
        println!("{}", mesh);
    }
}
