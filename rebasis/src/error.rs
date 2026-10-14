use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum Error {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("singular Gram matrix: {0}")]
    Singular(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::InvalidInput(msg.into()))
}
