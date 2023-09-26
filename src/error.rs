use thiserror::Error;


#[derive(Error, Debug)]
pub enum ModelError {
    #[error("{msg}, expected dim is: {expected:?}, got dim is: {got:?}")]
    UnexpectedDim {
        msg: String,
        expected: usize,
        got: usize,
    },
}