use itertools::Itertools;
use ndarray::prelude::*;
use std::collections::HashSet;

const DIRECTIONS: [isize; 3] = [1, 0, -1];

fn main() {
    let game = dbg!(GameState::new(&[6,7,4]));
}

#[derive(Debug)]
struct GameState {
    board: Array<Option<Color>, IxDyn>,
    players: [Player; 2],
    check_vecs: HashSet<Array<isize, Ix1>>,
}

impl GameState {
    pub fn new(dims: &[usize]) -> Self {
        GameState {
            board: Array::from_elem(dims, None),
            players: GameState::default_players(),
            check_vecs: GameState::generate_check_vecs(dims.len()),
        }
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
