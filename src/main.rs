mod err;

use err::GameError;
use itertools::Itertools;
use ndarray::prelude::*;
use std::collections::HashSet;

const POSITION_CHANGES: [isize; 3] = [1, 0, -1];

fn main() {
    let mut game = GameState::new(&[7, 6]).unwrap();
    game.play_disk(Color::Red, &mut vec![5]).unwrap();
}

#[derive(Debug, Clone, PartialEq)]
pub struct GameState {
    board: ArrayD<Option<Color>>,
    check_vecs: HashSet<Array1<isize>>,
    round: usize,
}

impl GameState {
    pub fn new(dims: &[usize]) -> Result<Self, GameError> {
        if dims.len() < 2 {
            return Err(GameError::TooFewDimensions);
        }

        Ok(GameState {
            board: Array::from_elem(dims, None),
            check_vecs: GameState::generate_check_vecs(dims.len()),
            round: 1,
        })
    }

    /// Insert a disk, actually a hypersphere, into the board and check if this wins the game.
    ///
    /// A disk of the given `color` will be inserted at the position specified in `pos`.
    /// This position has to specify the exact index of all but one dimensions.
    /// The final position is the first free space along the unspecified axis.
    /// `pos` is updated to point exactly to the newly added disk.
    /// If there is no space left along this axis `Err(GameError::AxisFull)` will be returned.
    /// 
    /// Next this new position is checked for a winning row and the result returned.
    /// If the game has not been won, the round counter is incremented by one.
    pub fn play_disk(&mut self, color: Color, mut pos: &mut Vec<usize>) -> Result<bool, GameError> {
        if self.round > self.board.len() {
            return Err(GameError::BoardFull);
        }

        self.check_input(&pos);
        self.insert_disk(color, &mut pos)?;

        let win = self.is_win_position(color, &pos);
        if !win {
            self.round += 1;
        }

        Ok(win)
    }

    /// Insert a disk, actually a hypersphere, into the board.
    ///
    /// A disk of the given `color` will be inserted at the position specified in `pos`.
    /// This position has to specify the exact index of all but one dimensions.
    /// The final position is the first free space along the unspecified axis.
    /// `pos` will be updated to point exactly to the newly added disk.
    /// If there is no space left along this axis `Err(GameError::AxisFull)` will be returned.
    fn insert_disk(&mut self, color: Color, pos: &mut Vec<usize>) -> Result<(), GameError> {
        let index = self.index_from_pos(&pos);
        let slice: ndarray::SliceInfo<_, IxDyn> = ndarray::SliceInfo::new(&index).unwrap();
        let column = self.board.slice_mut(slice.as_ref());

        if let Some((i, elem)) = column.into_iter().find_position(|elem| elem.is_none()) {
            *elem = Some(color);
            pos.push(i);
            return Ok(());
        }

        Err(GameError::AxisFull)
    }

    fn is_win_position(&self, color: Color, pos: &[usize]) -> bool {
        let pos = Array1::from(pos.to_owned()).map(|&p| p as isize);

        for direction in &self.check_vecs {
            let mut score = 1; // the starting position is already included
            let mut checking = pos.to_owned();

            // count the disks with `color` in the given `direction` and add them to the score
            score += self.check_direction(color, &mut checking, direction);

            if score >= 4 {
                // checking in one direction can be enough
                return true;
            }

            // now do the same in the opposite direction
            checking = pos.to_owned();
            score += self.check_direction(color, &mut checking, &-direction);
            if score >= 4 {
                return true;
            }
        }

        false
    }

    fn check_direction(
        &self,
        color: Color,
        starting_pos: &mut Array1<isize>,
        direction: &Array1<isize>,
    ) -> usize {
        let mut score = 0;

        for _ in 0..3 {
            *starting_pos += direction;

            let out_of_bounds = starting_pos
                .indexed_iter()
                .any(|(i, &ind)| ind < 0 || ind >= self.board.dim()[i] as isize);

            if out_of_bounds
                || !(Some(color)
                    == self.board[starting_pos.map(|&i| i as usize).as_slice().unwrap()])
            {
                // First condition: The calculated position is outside the board, so there is
                //                  nothing more to check in this direction.
                // Second condition: Stop looking in this direction if a field not containing a
                //                   disk of the given color is found
                break;
            }

            score += 1;
        }

        score
    }

    pub fn max_rounds(&self) -> usize {
        self.board.len()
    }

    pub const fn current_round(&self) -> usize {
        self.round
    }

    pub const fn disks_played(&self) -> usize {
        self.current_round() - 1
    }

    fn check_input(&self, pos: &[usize]) {
        if pos.len() != self.board.ndim() - 1 {
            panic!("The input position has to specify the coordinates in {} dimensions, but {} were given",
                self.board.ndim() - 1,
                pos.len()
            );
        }
    }

    fn index_from_pos(&self, pos: &[usize]) -> Vec<ndarray::SliceOrIndex> {
        let mut index = pos
            .iter()
            .map(|&ind| ndarray::SliceOrIndex::from(ind))
            .collect::<Vec<ndarray::SliceOrIndex>>();

        for _ in index.len()..self.board.raw_dim().ndim() {
            index.push(ndarray::SliceOrIndex::from(..));
        }

        index
    }

    /// Generate the direction vectors needed to check if a position is part of a winning row.
    fn generate_check_vecs(n_dims: usize) -> HashSet<Array<isize, Ix1>> {
        // The number of vectors which will be stored.
        // `[-1, 0, 1]` are the possible directions per dimension.
        // This means there are `3^n_dims` possible combinations.
        // `[0, 0, 0]` is removed since it doesn't change the position.
        // Each vector needs to be associated with its inverse, so we only keep one of them,
        // which further cuts down the number of vectors by half.
        let n_vecs = (3usize.pow(n_dims as u32) - 1) / 2;
        let mut directions: HashSet<Array<isize, Ix1>> = HashSet::with_capacity(n_vecs);

        // Generate the direction vectors and add them to the set one by one,
        // each time checking if their inverse is already part of the set.
        (0..n_dims)
            .map(|_| POSITION_CHANGES.iter().cloned())
            .multi_cartesian_product()
            .map(Array::from)
            .for_each(|vec| {
                if !directions.contains(&-&vec) {
                    directions.insert(vec);
                }
            });

        // Remove the vector only containing zeros.
        directions.remove(&Array::from_elem(n_dims, 0));
        directions
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Color {
    Red,
    Yellow,
}

#[cfg(test)]
mod tests {
    use super::{Color, GameError, GameState};

    const N_DIMS: usize = 10;
    const DIMS: [usize; N_DIMS] = [3; N_DIMS];

    #[test]
    fn create_with_less_than_two_dimensions() {
        assert_eq!(GameState::new(&[]), Err(GameError::TooFewDimensions));
        assert_eq!(GameState::new(&[6]), Err(GameError::TooFewDimensions));
    }

    #[test]
    fn create_with_up_to_ten_dimensions() {
        for i in 1..N_DIMS - 1 {
            assert!(GameState::new(&DIMS[0..=i]).is_ok());
        }
    }

    #[test]
    fn d2_board_full() {
        let mut game = GameState::new(&DIMS[0..2]).unwrap();

        // fill the board with yellow disks
        // the game can't finish, since no dimension is longer than 3
        for i in 0..game.max_rounds() {
            assert_eq!(game.play_disk(Color::Yellow, &mut vec![i % 3]), Ok(false));
        }

        // check every possible input location
        for i in 0..3 {
            assert_eq!(
                game.play_disk(Color::Yellow, &mut vec![i]),
                Err(GameError::BoardFull)
            );
        }
    }

    #[test]
    fn d2_big_board_full() {
        let x = 1001;
        let y = 1000;
        let mut game = GameState::new(&[x, y]).unwrap();

        // fill the board
        let mut last_played = Color::Red;
        let mut cur_color = Color::Yellow;

        for i in 0..game.max_rounds() {
            assert_eq!(game.play_disk(cur_color, &mut vec![i * 2 % x]), Ok(false));
            std::mem::swap(&mut last_played, &mut cur_color);
        }

        // check every possible input location
        for i in 0..y {
            assert_eq!(
                game.play_disk(Color::Yellow, &mut vec![i]),
                Err(GameError::BoardFull)
            );
        }
    }
}
