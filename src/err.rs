#[derive(Debug, Clone, PartialEq)]
pub enum GameError {
    BoardFull,
    AxisFull,
    TooFewDimensions,
}
