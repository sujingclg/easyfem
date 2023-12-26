mod edge;
mod quad;

use core::f64;

use crate::base::{elements::ElementBase, gauss::Gauss};

pub trait PoissonElement<const N: usize>: ElementBase<N> {
    fn poisson_stiffness_calc(&mut self, gauss: &Gauss, f: f64);
}
