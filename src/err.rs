#[derive(Debug, Clone, PartialEq)]
pub enum GameError {
    BoardFull,
    ColumnFull,
    TooFewDimensions,
}
