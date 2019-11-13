#[allow(clippy::all)]
use clap::{App, Arg};
use image::imageops::grayscale;
use image::{self, ImageBuffer, Luma};
use std::convert::TryInto;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::u8;

fn median_threshold(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let mut values: Vec<_> = img.pixels().map(|Luma([val])| val).collect();
    values.sort();
    let median = values[values.len() / 2];
    ImageBuffer::from_fn(width, height, |x, y| {
        let Luma([val]) = img[(x, y)];
        if val <= *median {
            Luma([0])
        } else {
            Luma([255])
        }
    })
}

fn average_threshold(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let total: f64 = img.pixels().map(|Luma([val])| f64::from(*val)).sum::<f64>();
    let average: f64 = total / img.len() as f64;
    let average: u8 = average.trunc() as u8;
    ImageBuffer::from_fn(width, height, |x, y| {
        let Luma([val]) = img[(x, y)];
        if val <= average {
            Luma([0])
        } else {
            Luma([255])
        }
    })
}

#[cfg(old)]
fn dilate(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    structuring_element: [[bool; 3]; 3],
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let mut buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    let full_offset = structuring_element.len();
    let offset = full_offset / 2;
    for y in 0..height {
        for x in 0..width {
            let mut local_max = u8::MIN;
            let (x_min, el_x_min) = if x < offset as u32 {
                (0, offset - x as usize)
            } else {
                (x as usize - offset, 0)
            };
            let (y_min, el_y_min) = if y < offset as u32 {
                (0, offset - y as usize)
            } else {
                (y as usize - offset, 0)
            };
            let (x_max, el_x_max) = if x + offset as u32 >= width {
                (
                    width as usize,
                    (x + offset as u32 - width + 1).try_into().unwrap(),
                )
            } else {
                (x as usize + 1 + offset, full_offset)
            };
            let (y_max, el_y_max) = if y + offset as u32 >= height {
                (
                    height as usize,
                    (y + offset as u32 - height + 1).try_into().unwrap(),
                )
            } else {
                (y as usize + 1 + offset, full_offset)
            };

            for (orig_y, i) in (y_min..y_max).zip(el_y_min..el_y_max) {
                for (orig_x, j) in (x_min..x_max).zip(el_x_min..el_x_max) {
                    if structuring_element[i][j] {
                        let image::Luma([current_val]) =
                            img[(orig_x.try_into().unwrap(), orig_y.try_into().unwrap())];
                        local_max = local_max.max(current_val);
                    }
                }
            }
            buffer[(x, y)] = image::Luma([local_max]);
        }
    }
    buffer
}

#[cfg(not(old))]
fn dilate(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    structuring_element: [[bool; 3]; 3],
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    fold_over_structuring_element(
        img,
        structuring_element,
        u8::MIN,
        |local_max, current_val| local_max.max(current_val),
    )
}

#[cfg(old)]
fn erode(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    structuring_element: [[bool; 3]; 3],
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let mut buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    let full_offset = structuring_element.len();
    let offset = full_offset / 2;
    for y in 0..height {
        for x in 0..width {
            let mut local_min = u8::MAX;

            // Determine the start and end indices for the image and
            // the structuring element, in order to avoid out-of-bounds
            // panics.
            let (x_min, el_x_min) = if x < offset as u32 {
                (0, offset - x as usize)
            } else {
                (x as usize - offset, 0)
            };
            let (y_min, el_y_min) = if y < offset as u32 {
                (0, offset - y as usize)
            } else {
                (y as usize - offset, 0)
            };
            let (x_max, el_x_max) = if x + offset as u32 >= width {
                (
                    width as usize,
                    (x + offset as u32 - width + 1).try_into().unwrap(),
                )
            } else {
                (x as usize + 1 + offset, full_offset)
            };
            let (y_max, el_y_max) = if y + offset as u32 >= height {
                (
                    height as usize,
                    (y + offset as u32 - height + 1).try_into().unwrap(),
                )
            } else {
                (y as usize + 1 + offset, full_offset)
            };

            for (orig_y, i) in (y_min..y_max).zip(el_y_min..el_y_max) {
                for (orig_x, j) in (x_min..x_max).zip(el_x_min..el_x_max) {
                    if structuring_element[i][j] {
                        let image::Luma([current_val]) =
                            img[(orig_x.try_into().unwrap(), orig_y.try_into().unwrap())];
                        local_min = local_min.min(current_val);
                    }
                }
            }
            buffer[(x, y)] = image::Luma([local_min]);
        }
    }
    buffer
}

#[cfg(not(old))]
fn erode(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    structuring_element: [[bool; 3]; 3],
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    fold_over_structuring_element(
        img,
        structuring_element,
        u8::MAX,
        |local_min, current_val| local_min.min(current_val),
    )
}

fn dilate_sub_erode(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    structuring_element: [[bool; 3]; 3],
) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let mut buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    let dilated = dilate(&img, structuring_element);
    let eroded = erode(&img, structuring_element);

    for y in 0..height {
        for x in 0..width {
            let Luma([dilated_val]) = dilated[(x, y)];
            let Luma([eroded_val]) = eroded[(x, y)];
            buffer[(x, y)] = Luma([dilated_val - eroded_val]);
        }
    }
    buffer
}

fn fold_over_structuring_element<F>(
    img: &ImageBuffer<Luma<u8>, Vec<u8>>,
    structuring_element: [[bool; 3]; 3],
    initial_value: u8,
    f: F,
) -> ImageBuffer<Luma<u8>, Vec<u8>>
where
    F: Fn(u8, u8) -> u8,
{
    let (width, height) = img.dimensions();
    let mut buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    let full_offset = structuring_element.len();
    let offset = full_offset / 2;
    for y in 0..height {
        for x in 0..width {
            let (x_min, el_x_min) = if x < offset as u32 {
                (0, offset - x as usize)
            } else {
                (x as usize - offset, 0)
            };
            let (y_min, el_y_min) = if y < offset as u32 {
                (0, offset - y as usize)
            } else {
                (y as usize - offset, 0)
            };
            let (x_max, el_x_max) = if x + offset as u32 >= width {
                (
                    width as usize,
                    (x + offset as u32 - width + 1).try_into().unwrap(),
                )
            } else {
                (x as usize + 1 + offset, full_offset)
            };
            let (y_max, el_y_max) = if y + offset as u32 >= height {
                (
                    height as usize,
                    (y + offset as u32 - height + 1).try_into().unwrap(),
                )
            } else {
                (y as usize + 1 + offset, full_offset)
            };

            let mut acc = initial_value;
            for (orig_y, i) in (y_min..y_max).zip(el_y_min..el_y_max) {
                for (orig_x, j) in (x_min..x_max).zip(el_x_min..el_x_max) {
                    if structuring_element[i][j] {
                        let image::Luma([current_val]) =
                            img[(orig_x.try_into().unwrap(), orig_y.try_into().unwrap())];
                        acc = f(acc, current_val);
                    }
                }
            }
            buffer[(x, y)] = image::Luma([acc]);
        }
    }
    buffer
}

fn main() {
    let structuring_element = [[true; 3]; 3];
    let matches = App::new("imageproc")
        .arg(
            Arg::with_name("threshold")
                .short("-t")
                .long("--threshold")
                .multiple(false)
                .required(false)
                .takes_value(true)
                .possible_values(&["average", "median"])
                .help("thresholds the image before applying operations"),
        )
        .arg(
            Arg::with_name("outfile")
                .short("-o")
                .long("--outfile")
                .takes_value(true)
                .required(false)
                .help("file to write the result to (defaults to <FILE>_processed.png)"),
        )
        .arg(
            Arg::with_name("OP")
                .required(true)
                .possible_values(&["dilate", "erode", "open", "close", "dilate_sub_erode"])
                .help("which image processing operation to apply to the image"),
        )
        .arg(
            Arg::with_name("FILE")
                .required(true)
                .help("path to file to process"),
        )
        .get_matches();

    let filepath = matches
        .value_of("FILE")
        .map(Path::new)
        .expect("no filename entered");
    let op = matches.value_of("OP").expect("no operation specified");

    let start_time = Instant::now();

    let img = image::open(filepath)
        .as_ref()
        .map(grayscale)
        .map(|x| match matches.value_of("threshold") {
            Some("average") => average_threshold(&x),
            Some("median") => median_threshold(&x),
            _ => x,
        })
        .unwrap();
    img.save_with_format("kitty_gray.png", image::ImageFormat::PNG)
        .expect("could not save image");
    let processed_img = match op {
        "dilate" => dilate(&img, structuring_element),
        "erode" => erode(&img, structuring_element),
        "open" => dilate(&erode(&img, structuring_element), structuring_element),
        "close" => erode(&dilate(&img, structuring_element), structuring_element),
        "dilate_sub_erode" => dilate_sub_erode(&img, structuring_element),
        _ => panic!("unknown op specified"),
    };

    let outpath = matches.value_of_os("outfile").map_or_else(
        || {
            let mut outfile = filepath.file_stem().unwrap().to_os_string();
            outfile.push("_processed");
            PathBuf::from(
                filepath
                    .with_file_name(outfile)
                    .with_extension(filepath.extension().unwrap()),
            )
        },
        PathBuf::from,
    );
    processed_img.save(outpath).expect("could not save image");

    let end_time = Instant::now();
    println!("Elapsed time: {} ms", (end_time - start_time).as_millis());
}
