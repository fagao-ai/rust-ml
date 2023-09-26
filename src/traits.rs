use candle_core::Tensor;
use crate::error::ModelError;

pub trait ModelTrain {
    fn fit(&self, data_x: &Tensor, data_y: &Tensor) -> Result<(), ModelError>;
    fn predict(&self, data: &Tensor) -> Tensor;
}
