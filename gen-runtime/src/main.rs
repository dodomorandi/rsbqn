use std::{
    ffi::OsStr,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;

type BoxError = Box<dyn std::error::Error>;

fn cmd_sync(cmd: &str, args: &[&OsStr]) -> Result<Vec<u8>, BoxError> {
    let output = Command::new(cmd).args(args).output()?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        match output.status.code() {
            Some(code) => Err(format!(
                "unable to run command {cmd} with args {args:?}, process exited with status {code}"
            )
            .into()),
            None => Err(
                "unable to run command {cmd} with args {args:?}, process exited with status \
                 unknown"
                    .into(),
            ),
        }
    }
}

fn test(repo: &Path, test: &str) -> Result<Vec<u8>, BoxError> {
    let cwd = std::env::current_dir()?;
    let crs_path = cwd.join("crs.bqn");
    let cmd = crs_path.to_str().unwrap();
    let args = [repo.as_os_str(), test.as_ref()];
    let output = cmd_sync(cmd, &args)?;
    Ok(output)
}

#[derive(Parser)]
struct Cli {
    /// The path of the cloned BQN repository from https://github.com/mlochbaum/BQN
    bqn: PathBuf,

    /// The path of the generated Rust file.
    #[clap(short, long, default_value = "src/runtime.rs")]
    output: PathBuf,
}

fn main() -> Result<(), BoxError> {
    let cli = Cli::parse();

    let r0 = test(&cli.bqn, "r0")?;
    let r1 = test(&cli.bqn, "r1")?;
    let c = test(&cli.bqn, "c")?;
    let f = test(&cli.bqn, "f")?;

    let mut file = BufWriter::new(File::create(&cli.output)?);
    writeln!(file, "use core::f64::{{INFINITY,NEG_INFINITY}};")?;
    writeln!(
        file,
        "use crate::schema::{{Code,new_scalar,new_char,new_string,Bodies,A,V}};",
    )?;
    writeln!(file, "use bacon_rajan_cc::Cc;")?;
    writeln!(
        file,
        "pub fn r0(provide: &A) -> Cc<Code> {{\n    Code::new({})\n}}\n",
        String::from_utf8_lossy(&r0)
    )?;
    writeln!(
        file,
        "pub fn r1(provide: &A,runtime_0v: &V) -> Cc<Code> {{\n\
        let runtime_0 = runtime_0v.as_a().unwrap();\n\
        Code::new({})\n}}\n\n",
        String::from_utf8_lossy(&r1)
    )?;
    writeln!(
        file,
        "pub fn c(runtimev: &V) -> Cc<Code> {{\n\
        let runtime = runtimev.as_a().unwrap();\n\
        Code::new({})\n}}\n\n",
        String::from_utf8_lossy(&c)
    )?;
    writeln!(
        file,
        "pub fn f(runtimev: &V) -> Cc<Code> {{\n\
        let runtime = runtimev.as_a().unwrap();\n
        Code::new({})\n}}\n\n",
        String::from_utf8_lossy(&f)
    )?;

    file.flush()?;
    Ok(())
}
