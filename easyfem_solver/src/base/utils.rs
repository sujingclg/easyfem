use std::{
    fs::File,
    io::{self, BufRead},
    path::Path,
};

pub fn read_lines<P>(file_path: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(file_path)?;
    Ok(io::BufReader::new(file).lines())
}

/// [10,20,30,40] => [10,11,20,21,30,31,40,41]
pub fn flatten_vector(v: &[usize], repeat_times: usize) -> Vec<usize> {
    v.iter()
        .flat_map(|x| {
            let mut vec = Vec::with_capacity(repeat_times);
            for i in 0..repeat_times {
                vec.push(*x + i);
            }
            vec
        })
        .collect()
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
        assert_eq!(flattened_2, [10, 11, 20, 21, 30, 31, 40, 41]);

        let flattened_3 = flatten_vector(&origin, 3);
        assert_eq!(
            flattened_3,
            [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]
        );
    }
}
