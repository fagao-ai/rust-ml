use crate::traits::ModelTrain;
use candle_core::{Device, Error, Shape, Tensor};
use crate::error::ModelError;

pub struct LinearRegression {
    w_matrix: Tensor,
    epochs: i8,
    eta: f32,
    batch_size: i8,
}

impl LinearRegression {
    fn new(n_features: usize, epochs: i8, eta: f32, batch_size: i8) -> Self {
        let device = Device::Cpu;
        let mut w_matrix = Tensor::rand(0f32, 1.0, (n_features, 1), &device).unwrap();
        Self {
            w_matrix,
            epochs,
            eta,
            batch_size,
        }
    }
}

impl ModelTrain for LinearRegression {
    fn fit(&self, data_x: &Tensor, data_y: &Tensor) -> Result<(), ModelError> {
        let data_x_dim_len = data_x.shape().dims().len();
        let data_y_dim_len = data_y.shape().dims().len();
        if data_x_dim_len != 2 {
            return Err(ModelError::UnexpectedDim {
                msg: "input data dims not be except dims".to_string(),
                expected: 2,
                got: data_x_dim_len,
            });
        }


        Ok(())
    }

fn predict(&self, data: &Tensor) -> Tensor {
    data.clone()
}
}
