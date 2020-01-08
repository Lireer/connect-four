mod err;

use err::GameError;
use itertools::Itertools;
use ndarray::prelude::*;
use std::collections::HashSet;

const DIRECTIONS: [isize; 3] = [1, 0, -1];

fn main() {
    let mut game = GameState::new(&[7, 6]).unwrap();
    game.play_disk(Color::Red, vec![5]).unwrap();
}

#[derive(Debug)]
struct GameState {
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

    pub fn play_disk(&mut self, color: Color, mut pos: Vec<usize>) -> Result<bool, GameError> {
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

    fn insert_disk(&mut self, color: Color, pos: &mut Vec<usize>) -> Result<(), GameError> {
        let index = self.index_from_pos(&pos);
        let slice: ndarray::SliceInfo<_, IxDyn> = ndarray::SliceInfo::new(&index).unwrap();
        let column = self.board.slice_mut(slice.as_ref());

        if let Some((i, elem)) = column.into_iter().find_position(|elem| elem.is_none()) {
            *elem = Some(color);
            pos.push(i);
            return Ok(());
        }

        Err(GameError::ColumnFull)
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
                // Second condition: Anything else in this direction will not add to the score
                break;
            }

            score += 1;
        }

        score
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
            .map(|_| DIRECTIONS.iter().cloned())
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
enum Color {
    Red,
    Yellow,
}
