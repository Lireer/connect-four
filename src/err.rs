#[derive(Debug)]
pub enum GameError {
    BoardFull,
    ColumnFull,
    TooFewDimensions,
}
