mod edge;
mod quad;

use core::f64;

use crate::base::{gauss::Gauss, primitives::PrimitiveBase};

/// N -> 单元节点个数
/// D -> 坐标系维度
pub trait PoissonElement<const N: usize, const D: usize>: PrimitiveBase<N> {
    fn poisson_stiffness_calc(&mut self, gauss: &impl Gauss<N, D>, f: f64);
}
