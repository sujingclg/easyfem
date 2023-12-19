// use nalgebra::{Dyn, Matrix, VecStorage, U1};

// pub type DMatrixf64 = Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>;
// pub type DMatrixUsize = Matrix<usize, Dyn, Dyn, VecStorage<usize, Dyn, Dyn>>;
// pub type DVectorf64 = Matrix<f64, Dyn, U1, VecStorage<f64, Dyn, U1>>;

pub mod base;
#[allow(non_snake_case)]
pub mod elements;
pub mod file_readers;
#[allow(non_snake_case)]
pub mod materials;
#[allow(non_snake_case)]
pub mod solvers;

pub fn run() {
    println!("run lib");
}
