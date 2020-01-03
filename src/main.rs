mod err;

use err::GameError;
use itertools::Itertools;
use ndarray::prelude::*;
use std::collections::HashSet;

const DIRECTIONS: [isize; 3] = [1, 0, -1];

fn main() {
    let mut game = GameState::new(&[6, 7, 4, 4]).unwrap();
    game.play_disk(Color::Red, &[1, 2, 2]);
}

#[derive(Debug)]
struct GameState {
    board: ArrayD<Option<Color>>,
    players: [Player; 2],
    check_vecs: HashSet<Array1<isize>>,
    round: usize,
    dimensions: usize,
}

impl GameState {
    pub fn new(dims: &[usize]) -> Result<Self, GameError> {
        if dims.len() < 2 {
            return Err(GameError::TooFewDimensions);
        }

        Ok(GameState {
            board: Array::from_elem(dims, None),
            players: GameState::default_players(),
            check_vecs: dbg!(GameState::generate_check_vecs(dims.len())),
            round: 1,
            dimensions: dims.len(),
        })
    }

    pub fn play_disk(&mut self, color: Color, pos: &[usize]) -> Result<bool, GameError> {
        self.check_input(pos);
        let index = self.index_from_pos(pos);

        let disk_pos = dbg!(self.insert_disk(color, index)?);
        let win = self.is_win_position(color, &disk_pos);
        if !win {
            self.round += 1;
        }
        Ok(win)
    }

    fn insert_disk(
        &mut self,
        color: Color,
        mut insert_pos: Vec<ndarray::SliceOrIndex>,
    ) -> Result<Vec<ndarray::SliceOrIndex>, GameError> {
        let slice: ndarray::SliceInfo<_, IxDyn> = ndarray::SliceInfo::new(&insert_pos).unwrap();
        let column = self.board.slice_mut(slice.as_ref());

        if let Some((i, elem)) = column.into_iter().find_position(|elem| elem.is_none()) {
            *elem = Some(color);
            if let Some(ind) = insert_pos.last_mut() {
                *ind = ndarray::SliceOrIndex::from(i);
                return Ok(insert_pos);
            }
        }

        Err(GameError::ColumnFull)
    }

    fn is_win_position(&self, color: Color, pos: &[ndarray::SliceOrIndex]) -> bool {
        unimplemented!()
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

        for _ in index.len()..self.dimensions {
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

    fn default_players() -> [Player; 2] {
        [Player::new(Color::Red), Player::new(Color::Yellow)]
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Color {
    Red,
    Yellow,
}

#[derive(Debug)]
struct Player {
    color: Color,
}

impl Player {
    pub fn new(color: Color) -> Self {
        Player { color }
    }
}
