mod edge;
mod quad;

use core::f64;

use crate::base::{elements::ElementBase, gauss::Gauss};

pub trait PoissonElement<const N: usize, const D: usize>: ElementBase<N> {
    fn poisson_stiffness_calc(&mut self, gauss: &impl Gauss<N, D>, f: f64);
}
