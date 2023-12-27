mod edge;
mod quad;

use nalgebra::DVector;

use crate::base::{gauss::Gauss, primitives::PrimitiveBase};

/// N -> 单元节点个数
/// D -> 坐标系维度
pub trait DiffusionElement<const N: usize, const D: usize>: PrimitiveBase<N, D> {
    fn diffusion_stiffness_calc(
        &mut self,
        gauss: &impl Gauss<N, D>,
        diffusivity: f64,
        dt: f64,
        prev_solution: &DVector<f64>,
    );
}
