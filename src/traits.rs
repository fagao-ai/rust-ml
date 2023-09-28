use candle_core::Tensor;
use crate::general_ml::common_model::Dataset;

pub trait ModelTrain {
    fn fit(&self,batch_idx: usize, data: Dataset) -> Result<(), Box<dyn std::error::Error>> ;
    fn predict(&self, data: &Tensor) -> Tensor;
}
