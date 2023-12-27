use gauss_quad::GaussLegendre;
use nalgebra::SMatrix;

/// 计算jacob行列式
pub fn jacob_determinant<const N: usize>(
    shp_grad: &SMatrix<f64, N, 1>,
    nodes_coordinates: &SMatrix<f64, N, 1>,
) -> f64 {
    let mut dx_dxi = 0.0;
    for i in 0..N {
        dx_dxi += shp_grad[i] * nodes_coordinates[i];
    }
    dx_dxi
}

/// 获取1D高斯矩阵, 行数是高斯点个数, 第一列是高斯权重w, 第二列是高斯点坐标xi
pub fn gauss_1d_vector(deg: usize) -> Vec<(f64, [f64; 1])> {
    let mut gauss_vector = vec![];
    // for every columns of the above matrix, 0->weight, 1->xi
    let gauss_legendre = GaussLegendre::init(deg);
    for i in 0..deg {
        gauss_vector.push((gauss_legendre.weights[i], [gauss_legendre.nodes[i]]));
    }
    gauss_vector
}

/// 获取2D高斯矩阵, 行数是高斯点个数, 第一列是高斯权重w, 第二列是高斯点坐标xi, 第三列是高斯点坐标eta
pub fn gauss_2d_vector(deg: usize) -> Vec<(f64, [f64; 2])> {
    let mut gauss_vector = vec![];
    // for every columns of the above matrix, 0->weight, 1->xi, 2->eta
    let gauss_legendre = GaussLegendre::init(deg);
    for i in 0..deg {
        for j in 0..deg {
            gauss_vector.push((
                gauss_legendre.weights[i] * gauss_legendre.weights[j],
                [gauss_legendre.nodes[i], gauss_legendre.nodes[j]],
            ));
        }
    }
    gauss_vector
}

/// 获取3D高斯矩阵, 行数是高斯点个数, 第一列是高斯权重w, 第二列是高斯点坐标xi, 第三列是高斯点坐标eta, 第四列是zeta
pub fn gauss_3d_vector(deg: usize) -> Vec<(f64, [f64; 3])> {
    let mut gauss_vector = vec![];
    // for every columns of the above matrix, 0->weight, 1->xi, 2->eta, 3->zeta
    let gauss_legendre = GaussLegendre::init(deg);
    for i in 0..deg {
        for j in 0..deg {
            for k in 0..deg {
                gauss_vector.push((
                    gauss_legendre.weights[i]
                        * gauss_legendre.weights[j]
                        * gauss_legendre.weights[k],
                    [
                        gauss_legendre.nodes[i],
                        gauss_legendre.nodes[j],
                        gauss_legendre.nodes[k],
                    ],
                ));
            }
        }
    }
    gauss_vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_1d_test() {
        let matrix = gauss_1d_vector(3);
        matrix.iter().for_each(|item| {
            println!("{:?}", item);
        });
    }

    #[test]
    fn gauss_2d_test() {
        let matrix = gauss_2d_vector(2);
        matrix.iter().for_each(|item| {
            println!("{:?}", item);
        });
    }

    #[test]
    fn gauss_3d_test() {
        let matrix = gauss_3d_vector(3);
        matrix.iter().for_each(|item| {
            println!("{:?}", item);
        });
    }
}
