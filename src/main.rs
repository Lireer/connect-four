use itertools::Itertools;
use ndarray::prelude::*;
use std::collections::HashSet;

const DIRECTIONS: [isize; 3] = [1, 0, -1];

fn main() {
    GameState::new(&[7,6]);
}

struct GameState {
    board: Array<Option<Color>, IxDyn>,
    players: [Player; 2],
    check_vecs: Array<i8, IxDyn>,
}

impl GameState {
    pub fn new(dims: &[usize]) {
        GameState::generate_check_vecs(dims.len());
    }

    /// Generate the direction vectors needed for the 
    fn generate_check_vecs(n_dims: usize) -> HashSet<Array<isize, Ix1>> {
        
        // The number of vectors which will be stored.
        // `[-1, 0, 1]` are the possible directions per dimension.
        // This means there are `3^n_dims` possible combinations.
        // `[0, 0, 0]` is removed since it doesn't change the position.
        // Each vector needs to be associated with its inverse, so we only keep one of them,
        // which further cuts down the number of vectors by half.
        let n_vecs = (3^n_dims-1)/2;
        let mut directions: HashSet<Array<isize, Ix1>> = HashSet::with_capacity(n_vecs);

        // Generate the direction vectors and add them to the set one by one,
        // each time checking if their inverse is already part of the set.
        (0..n_dims)
            .map(|_| DIRECTIONS.iter().cloned())
            .multi_cartesian_product()
            .map(|vec| Array::from(vec))
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

#[derive(Debug, Eq, PartialEq)]
enum Color {
    Red,
    Yellow,
}

#[derive(Debug)]
struct Player {
    color: Color,
}

// Zur überprüfung wird pro board ein Vec Indizeveränderern genutzt um alle möglichen Siegachsen zu überprüfen
// z.B.
//      d = 3 (3-dimensional)
//      Board dim = (7,6,4)
//      New disk pos = [3,2,2] (Indizierung beginnend mit 0)
//      Check Vec = [[1,0,0], //
//                   [0,1,0], // These are moves using only one axes each
//                   [0,0,1], //
//                   [1,1,0],   //
//                   [1,0,1],   // These are moves using two axes each
//                   [0,1,1],   //
//                   [1,1,1],      // This is a move using all three axis
//                   ...    ] // To complete the vec add another vec for each possible combination of negative and positive ones
//
// Anzahl der zu überprüfenden Vektoren: $$\sum_{i=1}^{d} (2^i\binom{d}{i}) $$
// Da aber sowieso immer in einer Geraden ausgehend von der aktuellen Position gesucht wird, sollte nur eine Hälfte der Vektoren
// gespeichert werden, deren Inverse die fehlende Hälfte bildet.
