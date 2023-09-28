use candle_nn::{Linear, VarBuilder};
use candle_core::{Tensor, Result};


pub fn linear_z(in_dim: usize, out_dim: usize, vs: VarBuilder) -> Result<Linear> {
    let ws = vs.get_with_hints((out_dim, in_dim), "weight", candle_nn::init::ZERO)?;
    let bs = vs.get_with_hints(out_dim, "bias", candle_nn::init::ZERO)?;
    Ok(Linear::new(ws, Some(bs)))
}


pub struct Dataset {
    pub train_x: Tensor,
    pub train_y: Tensor,
    pub test_x: Tensor,
    pub test_y: Tensor,
    pub labels: usize,
}