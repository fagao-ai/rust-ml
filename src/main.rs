mod general_ml;
mod traits;
mod error;

use candle_core::{Device, Tensor};
use general_ml::linear_regression::LinearRegression;

fn main() {
    let device = Device::Cpu;
    let w_matrix = Tensor::rand(0f32, 1.0, (10, 1), &device).unwrap();
    let w_matrix1 = Tensor::rand(0f32, 1.0, (10, 1), &device).unwrap();
    // println!("{:?}", w_matrix.shape().dims().len());
    // let c = w_matrix.matmul(&w_matrix1).unwrap();
    println!("{w_matrix}");
    println!("{w_matrix1}");
    let c = w_matrix.add(&w_matrix1).unwrap();
    println!("{c}");
}
