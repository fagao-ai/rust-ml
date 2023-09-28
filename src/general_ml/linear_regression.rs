use candle_core::{D, Device, DType, Error, Module, Tensor};
use candle_nn::{Linear, loss, ops, Optimizer, VarBuilder, VarMap};
use rand::prelude::*;

use crate::error::ModelError;
use crate::general_ml::common_model::Dataset;
use crate::traits::ModelTrain;

pub struct LinearRegression {
    model: Linear,
    epochs: usize,
    eta: f64,
    model_parameters: VarMap,
}

impl LinearRegression {
    fn new(n_features: usize, out_dim: usize, epochs: usize, eta: f64) -> Result<LinearRegression, Error> {
        let dev = Device::cuda_if_available(0)?;
        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = candle_nn::linear(n_features, out_dim, vs.pp("fc1"))?;
        Ok(Self { model, epochs, eta, model_parameters: varmap })
    }
}

impl Module for LinearRegression {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        self.model.forward(x)
    }
}

impl ModelTrain for LinearRegression {
    fn fit(&self, batch_size: usize, dataset: Dataset) -> Result<(), Box<dyn std::error::Error>> {
        let data_x_dim_len = dataset.train_x.shape().dims().len();
        let data_y_dim_len = dataset.train_y.shape().dims().len();
        if data_x_dim_len != 2 {
            return Err(Box::new(ModelError::UnexpectedDim {
                msg: "input data_x dims not be except dims".to_string(),
                expected: 2,
                got: data_x_dim_len,
            }));
        }

        if data_y_dim_len != 2 {
            return Err(Box::new(ModelError::UnexpectedDim {
                msg: "input data_y dims not be except dims".to_string(),
                expected: 2,
                got: data_x_dim_len,
            }));
        }

        let adamw_params = candle_nn::ParamsAdamW {
            lr: self.eta,
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(self.model_parameters.all_vars(), adamw_params)?;
        let mut batch_idxes = (0..data_x_dim_len).collect::<Vec<usize>>();
        for _epoch in 0..self.epochs {
            let mut sum_loss = 0f32;
            batch_idxes.shuffle(&mut thread_rng());
            for batch_idx in batch_idxes.iter() {
                let train_x = dataset.train_x.narrow(0, batch_idx * batch_size, batch_size)?;
                let train_y = dataset.train_y.narrow(0, batch_idx * batch_size, batch_size)?;
                let logits = self.model.forward(&train_x)?;
                let mse_loss = loss::mse(&logits, &train_y)?;
                let mse_loss_value = mse_loss.mean_all()?;
                println!("loss: {mse_loss_value}");
                opt.backward_step(&mse_loss)?;
                sum_loss += mse_loss.to_vec0::<f32>()?;
            }
        }
        let locked_data = self.model_parameters.data().lock().unwrap();

        // 访问 HashMap 并打印键值对
        for (key, value) in locked_data.iter() {
            println!("Key: {key}, Value: {value}");
        }

        Ok(())
    }

    fn predict(&self, data: &Tensor) -> Tensor {
        data.clone()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn generate_sample_data(n_features: usize, num_samples: usize) -> Dataset {
        let device = Device::Cpu;
        let train_x = Tensor::rand(0f32, 1.0, (num_samples, n_features), &device).unwrap();
        let noise = (Tensor::rand(0f32, 1.0, (num_samples, 1), &device).unwrap() * 0.1).unwrap();
        let true_weights = Tensor::rand(0f32, 1.0, &[n_features, 1], &device).unwrap();
        let true_weights = Tensor::new(&[[10f32]], &device).unwrap();
        let true_bias = 2.0;
        let train_y = ((train_x.matmul(&true_weights).unwrap().add(&noise)).unwrap() + true_bias).unwrap();


        let test_x = Tensor::rand(0f32, 1.0, (num_samples, n_features), &device).unwrap();
        let true_weights = Tensor::rand(0f32, 1.0, &[n_features, 1], &device).unwrap();
        println!("true_weights: {true_weights}");
        println!("true_bias: {true_bias}");

        let test_y = (train_x.matmul(&true_weights).unwrap() + true_bias).unwrap();
        Dataset { train_x, train_y, test_x, test_y, labels: 2 }
    }

    #[test]
    fn test_linear_regression() {
        let n_features = 1;
        let out_dim = 1;
        let epochs = 10000;
        let eta = 0.2;
        let batch_size: usize = 128;

        let model = LinearRegression::new(n_features, out_dim, epochs, eta)
            .expect("Failed to create LinearRegression model");


        // 创建示例数据集（随机生成的数据）
        let dataset = generate_sample_data(n_features, 1000);

        // 训练模型
        if let Err(err) = model.fit(batch_size, dataset) {
            panic!("Training error: {:?}", err);
        }

    }
}

