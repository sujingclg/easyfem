use std::{
    fs::File,
    io::{self, BufRead},
    path::Path,
    usize,
};

pub fn read_lines<P>(file_path: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(file_path)?;
    Ok(io::BufReader::new(file).lines())
}

/// [10,20,30,40] => [20,21,40,41,60,61,80,81]
pub fn flatten_vector(v: &[usize], repeat_times: usize) -> Vec<usize> {
    v.iter()
        .flat_map(|x| {
            let mut vec = Vec::with_capacity(repeat_times);
            for i in 0..repeat_times {
                vec.push(*x * repeat_times + i);
            }
            vec
        })
        .collect()
}

pub fn square_range(n: usize) -> Vec<(usize, usize)> {
    let mut vec = Vec::new();
    for i in 0..n {
        for j in 0..n {
            vec.push((i, j));
        }
    }
    vec
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn flatten_vector_test() {
        let origin = vec![10, 20, 30, 40];
        let flattened_1 = flatten_vector(&origin, 1);
        assert_eq!(flattened_1, [10, 20, 30, 40]);

        let flattened_2 = flatten_vector(&origin, 2);
        assert_eq!(flattened_2, [20, 21, 40, 41, 60, 61, 80, 81]);

        let flattened_3 = flatten_vector(&origin, 3);
        assert_eq!(
            flattened_3,
            [30, 31, 32, 60, 61, 62, 90, 91, 92, 120, 121, 122]
        );
    }

    #[test]
    fn square_range_test() {
        let result = square_range(1);
        assert_eq!(result, [(0, 0)]);

        let result = square_range(2);
        assert_eq!(result, [(0, 0), (0, 1), (1, 0), (1, 1)]);

        let result = square_range(3);
        assert_eq!(
            result,
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2)
            ]
        );
    }
}
