//! Compare two folders of exported crops, pixel for pixel.
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let a_dir = args.next().expect("usage: cropdiff <dir-a> <dir-b>");
    let b_dir = args.next().expect("usage: cropdiff <dir-a> <dir-b>");
    let mut names: Vec<_> = std::fs::read_dir(&a_dir)?
        .filter_map(|e| e.ok().map(|e| e.file_name()))
        .collect();
    names.sort();
    println!(
        "{:<36} {:>9} {:>10} {:>16}",
        "crop", "max diff", "mean diff", "pixels differing"
    );
    for name in names {
        let a = image::open(std::path::Path::new(&a_dir).join(&name))?.to_rgb8();
        let b = image::open(std::path::Path::new(&b_dir).join(&name))?.to_rgb8();
        if a.dimensions() != b.dimensions() {
            println!("{:<36} SIZE MISMATCH", name.to_string_lossy());
            continue;
        }
        let (mut max, mut sum, mut differing) = (0u8, 0u64, 0u64);
        for (pa, pb) in a.pixels().zip(b.pixels()) {
            let d = pa.0.iter().zip(pb.0.iter()).map(|(x, y)| x.abs_diff(*y));
            let mut any = false;
            for v in d {
                max = max.max(v);
                sum += u64::from(v);
                any |= v != 0;
            }
            if any {
                differing += 1;
            }
        }
        let total = u64::from(a.width()) * u64::from(a.height());
        println!(
            "{:<36} {max:>9} {:>10.4} {differing:>9} ({:>4.1}%)",
            name.to_string_lossy(),
            sum as f64 / (total * 3) as f64,
            100.0 * differing as f64 / total as f64
        );
    }
    Ok(())
}
