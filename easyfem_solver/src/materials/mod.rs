use nalgebra::{Matrix1, Matrix3, Matrix6, SMatrix};

/// 材料本构计算
/// N为本构矩阵的阶数
/// 和单元维度D的关系是 N = D*(D+1)/2
pub trait Material<const N: usize> {
    /// 获取材料本构矩阵的引用
    fn get_constitutive_matrix(&self) -> &SMatrix<f64, N, N>;
}

pub struct IsotropicLinearElastic1D {
    constitutive_matrix: Matrix1<f64>,
}

impl IsotropicLinearElastic1D {
    pub fn new(E: f64, A: f64) -> Self {
        IsotropicLinearElastic1D {
            constitutive_matrix: Matrix1::new(E * A),
        }
    }
}

impl Material<1> for IsotropicLinearElastic1D {
    fn get_constitutive_matrix(&self) -> &Matrix1<f64> {
        &self.constitutive_matrix
    }
}

pub enum PlaneCondition {
    PlaneStress,
    PlaneStrain,
}

pub struct IsotropicLinearElastic2D {
    constitutive_matrix: Matrix3<f64>,
}

impl IsotropicLinearElastic2D {
    pub fn new(E: f64, v: f64, condition: PlaneCondition, t: f64) -> Self {
        use PlaneCondition::*;
        match condition {
            PlaneStress => IsotropicLinearElastic2D {
                constitutive_matrix: Matrix3::new(
                    1.0,
                    v,
                    0.0,
                    v,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    (1.0 - v) / 2.0,
                ) * E
                    / (1.0 - v.powi(2))
                    * t,
            },
            PlaneStrain => IsotropicLinearElastic2D {
                constitutive_matrix: Matrix3::new(
                    1.0 - v,
                    v,
                    0.0,
                    v,
                    1.0 - v,
                    0.0,
                    0.0,
                    0.0,
                    (1.0 - 2.0 * v) / 2.0,
                ) * E
                    / (1.0 + v)
                    / (1.0 - 2.0 * v)
                    * t,
            },
        }
    }
}

impl Material<3> for IsotropicLinearElastic2D {
    fn get_constitutive_matrix(&self) -> &Matrix3<f64> {
        &self.constitutive_matrix
    }
}

pub struct IsotropicLinearElastic3D {
    constitutive_matrix: Matrix6<f64>,
}

impl IsotropicLinearElastic3D {
    pub fn new(E: f64, v: f64) -> Self {
        let mut constitutive_matrix = Matrix6::zeros();
        let factor = E / (1.0 + v) / (1.0 - 2.0 * v);
        constitutive_matrix[(0, 0)] = (1.0 - v) * factor;
        constitutive_matrix[(0, 1)] = v * factor;
        constitutive_matrix[(0, 2)] = v * factor;
        constitutive_matrix[(1, 0)] = v * factor;
        constitutive_matrix[(1, 1)] = (1.0 - v) * factor;
        constitutive_matrix[(1, 2)] = v * factor;
        constitutive_matrix[(2, 0)] = v * factor;
        constitutive_matrix[(2, 1)] = v * factor;
        constitutive_matrix[(2, 2)] = (1.0 - v) * factor;
        constitutive_matrix[(3, 3)] = (1.0 - 2.0 * v) / 2.0 * factor;
        constitutive_matrix[(4, 4)] = (1.0 - 2.0 * v) / 2.0 * factor;
        constitutive_matrix[(5, 5)] = (1.0 - 2.0 * v) / 2.0 * factor;

        IsotropicLinearElastic3D {
            constitutive_matrix,
        }
    }
}

impl Material<6> for IsotropicLinearElastic3D {
    fn get_constitutive_matrix(&self) -> &Matrix6<f64> {
        &self.constitutive_matrix
    }
}
