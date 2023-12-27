use nalgebra::SMatrix;

use super::{utils::gauss_3d_vector, Gauss, GaussResult};

/// N -> 单元节点个数
pub struct GaussCube<const N: usize> {
    gauss_vector: Vec<(f64, [f64; 3])>, // 高斯积分矩阵, 行数为高斯积分点个数
                                        // 1列->w 2列->xi 3列->eta 4列->zeta
}

impl<const N: usize> GaussCube<N> {
    pub fn new(gauss_deg: usize) -> Self {
        GaussCube {
            gauss_vector: gauss_3d_vector(gauss_deg),
        }
    }
}

pub type GaussCube8 = GaussCube<8>;

impl Gauss<8, 3> for GaussCube8 {
    fn gauss_vector(&self) -> &Vec<(f64, [f64; 3])> {
        &self.gauss_vector
    }

    fn shape_func_calc(
        &self,
        nodes_coordinates: &nalgebra::SMatrix<f64, 8, 3>, // 单元的节点坐标矩阵
        gauss_point: &[f64; 3],
    ) -> super::GaussResult<8, 3> {
        let [xi, eta, zeta] = gauss_point;
        // 3维8节点六面体等参元形函数
        let shp_val = SMatrix::<f64, 8, 1>::from_column_slice(&[
            (1.0 - xi) * (1.0 - eta) * (1.0 - zeta) / 8.0, // N1
            (1.0 + xi) * (1.0 - eta) * (1.0 - zeta) / 8.0, // N2
            (1.0 + xi) * (1.0 + eta) * (1.0 - zeta) / 8.0, // N3
            (1.0 - xi) * (1.0 + eta) * (1.0 - zeta) / 8.0, // N4
            (1.0 - xi) * (1.0 - eta) * (1.0 + zeta) / 8.0, // N5
            (1.0 + xi) * (1.0 - eta) * (1.0 + zeta) / 8.0, // N6
            (1.0 + xi) * (1.0 + eta) * (1.0 + zeta) / 8.0, // N7
            (1.0 - xi) * (1.0 + eta) * (1.0 + zeta) / 8.0, // N8
        ]);

        /*
         * 梯度矩阵，每行元素是每个形函数分别对xi、eta、zeta求偏导:
         *  第一列 \frac{\partial Ni}{\partial \xi}
         *  第二列 \frac{\partial Ni}{\partial \eta}
         *  第三列 \frac{\partial Ni}{\partial \zeta}
         */
        let mut shp_grad = SMatrix::<f64, 8, 3>::from_row_slice(&[
            // N1
            -(1.0 - eta) * (1.0 - zeta) / 8.0, // dN1/dxi
            -(1.0 - xi) * (1.0 - zeta) / 8.0,  // dN1/deta
            -(1.0 - xi) * (1.0 - eta) / 8.0,   // dN1/dzeta
            // N2
            (1.0 - eta) * (1.0 - zeta) / 8.0, // dN2/dxi
            -(1.0 + xi) * (1.0 - zeta) / 8.0, // dN2/deta
            -(1.0 + xi) * (1.0 - eta) / 8.0,  // dN2/dzeta
            // N3
            (1.0 + eta) * (1.0 - zeta) / 8.0, // dN3/dxi
            (1.0 + xi) * (1.0 - zeta) / 8.0,  // dN3/deta
            -(1.0 + xi) * (1.0 + eta) / 8.0,  // dN3/dzeta
            // N4
            -(1.0 + eta) * (1.0 - zeta) / 8.0, // dN4/xi
            (1.0 - xi) * (1.0 - zeta) / 8.0,   // dN4/deta
            -(1.0 - xi) * (1.0 + eta) / 8.0,   // dN4/dzeta
            // N5
            -(1.0 - eta) * (1.0 + zeta) / 8.0, // dN5/dxi
            -(1.0 - xi) * (1.0 + zeta) / 8.0,  // dN5/deta
            (1.0 - xi) * (1.0 - eta) / 8.0,    // dN5/dzeta
            // N6
            (1.0 - eta) * (1.0 + zeta) / 8.0, // dN6/dxi
            -(1.0 + xi) * (1.0 + zeta) / 8.0, // dN6/deta
            (1.0 + xi) * (1.0 - eta) / 8.0,   // dN6/dzeta
            // N7
            (1.0 + eta) * (1.0 + zeta) / 8.0, // dN7/dxi
            (1.0 + xi) * (1.0 + zeta) / 8.0,  // dN7/deta
            (1.0 + xi) * (1.0 + eta) / 8.0,   // dN7/dzeta
            // N8
            -(1.0 + eta) * (1.0 + zeta) / 8.0, // dN8/dxi
            (1.0 - xi) * (1.0 + zeta) / 8.0,   // dN8/deta
            (1.0 - xi) * (1.0 + eta) / 8.0,    // dN8/dzeta
        ]);

        /*
         * jacob矩阵形式为
         * [[dx_dxi, dy_dxi, dz_dxi],
         * [dx_deta, dy_deta, dz_deta],
         * [dx_dzeta, dy_dzeta, dz_dzeta]]
         */
        let jacob_matrix = shp_grad.transpose() * nodes_coordinates;
        let det_j = jacob_matrix.determinant();

        /*
         * jacob逆矩阵形式为
         * [[dxi_dx, deta_dx, dzeta_dx],
         * [dxi_dy, deta_dy, dzeta_dy],
         * [dxi_dz, deta_dz, dzeta_dz]]
         */
        let inverse_j = jacob_matrix.try_inverse().unwrap(); // 逆矩阵

        shp_grad = shp_grad * inverse_j.transpose();

        GaussResult {
            shp_val,
            shp_grad,
            det_j,
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    // #[test]
    // fn test() {}
}
