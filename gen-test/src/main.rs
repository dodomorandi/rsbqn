use std::{
    fmt,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter},
    ops::Not,
    path::{Path, PathBuf},
    process,
    str::FromStr,
};

use clap::Parser;

#[derive(Parser)]
struct Cli {
    /// The path of the cloned BQN repository from https://github.com/mlochbaum/BQN
    bqn: PathBuf,

    /// The path for the directory for the generated Rust files.
    #[clap(short, long, default_value = "./tests")]
    output: PathBuf,
}

type BoxError = Box<dyn std::error::Error>;

fn main() -> Result<(), BoxError> {
    let cli = Cli::parse();

    runtime_tests(&cli)?;
    compiler_tests(&cli)?;
    Ok(())
}

enum Template {
    Core,
    Compiler,
}

const fn template(ty: Template) -> &'static str {
    match ty {
        Template::Core => {
            "\
use log::{{info}};
use core::f64::{INFINITY,NEG_INFINITY};
use rsbqn::init_log;
use rsbqn::vm::{run,call,runtime,prog};
use rsbqn::schema::{Code,Env,new_scalar,new_char,new_string,Bodies,A,Decoder,V,Stack};\n"
        }
        Template::Compiler => {
            "\
use log::{info};
use core::f64::{INFINITY,NEG_INFINITY};
use rsbqn::init_log;
use rsbqn::vm::{run,call,runtime,prog};
use rsbqn::schema::{Code,Env,new_scalar,new_char,new_string,Bodies,A,Decoder,V,Stack};
use rsbqn::gen::code::c;
use bacon_rajan_cc::Cc;\n"
        }
    }
}

fn write_suite(bytecode: &[String], path: &Path, template_ty: Template) -> io::Result<()> {
    use std::io::Write;

    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "{}", template(template_ty))?;
    for line in bytecode {
        writeln!(file, "{line}")?;
    }
    file.flush()
}

fn runtime_test(name: &str, dependency: Dependency, cli: &Cli) -> Result<(), BoxError> {
    let bqn_name = format!("{name}.bqn");
    let test_name = format!("tests/{name}.rs");
    let code = suite(&bqn_name, name, dependency, true, cli)?;
    write_suite(&code, &PathBuf::from(test_name), Template::Core)?;
    Ok(())
}

fn runtime_tests(cli: &Cli) -> Result<(), BoxError> {
    runtime_test("bytecode", Dependency::Undefined, cli)?;
    runtime_test("simple", Dependency::Runtime, cli)?;
    runtime_test("prim", Dependency::Runtime, cli)?;
    runtime_test("undo", Dependency::Runtime, cli)?;
    runtime_test("identity", Dependency::Runtime, cli)?;
    Ok(())
}

fn compiler_test(name: &str, commend_mode: bool, cli: &Cli) -> Result<(), BoxError> {
    let bqn_name = format!("{name}.bqn");
    let test_name = format!("tests/{name}.rs");
    let code = suite(&bqn_name, name, Dependency::Compiler, commend_mode, cli)?;
    write_suite(&code, &PathBuf::from(test_name), Template::Compiler)?;
    Ok(())
}

fn compiler_tests(cli: &Cli) -> Result<(), BoxError> {
    compiler_test("literal", true, cli)?;
    compiler_test("syntax", true, cli)?;

    // token level tests require comment_mode=false
    compiler_test("token", false, cli)?;
    compiler_test("header", true, cli)?;
    compiler_test("fill", true, cli)?;
    compiler_test("namespace", true, cli)?;
    compiler_test("unhead", true, cli)?;
    Ok(())
}

fn suite(
    name: &str,
    short_name: &str,
    dependency: Dependency,
    comment_mode: bool,
    cli: &Cli,
) -> Result<Vec<String>, BoxError> {
    let path = cli.bqn.join("test").join("cases").join(name);
    let file = BufReader::new(File::open(path)?);
    let parsed = file
        .lines()
        .flat_map(|line| {
            let line = match line {
                Ok(x) => x,
                Err(err) => return Some(Err(err)),
            };
            parse_line(line, comment_mode).map(Ok)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let tests = gen_tests(cli, &parsed, dependency)?;
    Ok(gen_code(short_name, &tests, dependency))
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct Parsed {
    code: String,
    comment: Option<String>,
    expected: ExpectedValue,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum ExpectedValue {
    Number(ExpectedNumber),
    Assert,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
enum ExpectedNumber {
    Integer(i64),
    Float(f64),
}

impl fmt::Display for ExpectedNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpectedNumber::Integer(n) => n.fmt(f),
            ExpectedNumber::Float(n) => n.fmt(f),
        }
    }
}

#[derive(Debug)]
struct InvalidNumericValue;

impl FromStr for ExpectedValue {
    type Err = InvalidNumericValue;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.contains('.') {
            s.parse()
                .map(|n| Self::Number(ExpectedNumber::Float(n)))
                .map_err(|_| InvalidNumericValue)
        } else {
            s.parse()
                .map(|n| Self::Number(ExpectedNumber::Integer(n)))
                .map_err(|_| InvalidNumericValue)
        }
    }
}

fn parse_line(mut line: String, comment_mode: bool) -> Option<Parsed> {
    let expected_code_comment = if line.starts_with('!') {
        line.split_once('%')
            .map(|(_, code_comment)| (ExpectedValue::Assert, code_comment[1..].to_owned()))
    } else {
        line.starts_with('#').not().then(|| match line.find('%') {
            Some(expected_pos) => {
                let rest = line.split_off(expected_pos + 1);
                let expected = line[..line.len() - 1].trim().parse().unwrap();
                (expected, rest)
            }
            None => (ExpectedValue::Number(ExpectedNumber::Float(1.0)), line),
        })
    };

    expected_code_comment
        .map(|(expected, code_comment)| {
            let (code, comment) = split_comment(code_comment, comment_mode);
            Parsed {
                expected,
                code,
                comment,
            }
        })
        .filter(|parsed| parsed.code.trim().is_empty().not())
}

fn split_comment(mut line: String, comment_mode: bool) -> (String, Option<String>) {
    let comment = comment_mode
        .then(|| line.find('#'))
        .flatten()
        .map(|comment_position| {
            let rest = line.split_off(comment_position + 1);
            line.truncate(line.len() - 1);
            rest
        });
    (line, comment)
}

struct Test<'a> {
    parsed: &'a Parsed,
    bytecode: Option<String>,
}

#[derive(Debug)]
struct BqnCommandError(String);

impl fmt::Display for BqnCommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "error while running the BQN code \"{}\"", self.0)
    }
}

impl std::error::Error for BqnCommandError {}

fn gen_tests<'a>(
    cli: &Cli,
    parsed: &'a [Parsed],
    dependency: Dependency,
) -> Result<Vec<Test<'a>>, BoxError> {
    let bqn_file_path = cli.bqn.join("src").join("cjs.bqn");
    parsed
        .iter()
        .map(|parsed| {
            let bytecode = matches!(dependency, Dependency::Runtime | Dependency::Undefined)
                .then(|| {
                    let code = parsed.code.trim();
                    let output = process::Command::new(bqn_file_path.as_os_str())
                        .arg(code)
                        .output()
                        .map_err(BoxError::from)?;

                    if output.status.success() {
                        dbg!(String::from_utf8(output.stdout).map_err(BoxError::from))
                    } else {
                        Err(BoxError::from(BqnCommandError(code.to_string())))
                    }
                })
                .transpose()?;

            Ok(Test { parsed, bytecode })
        })
        .collect()
}

fn gen_code(name: &str, tests: &[Test], dependency: Dependency) -> Vec<String> {
    tests
        .iter()
        .enumerate()
        .map(|(index, test)| {
            let Test { parsed, bytecode } = test;
            gen_line(
                name,
                parsed.expected,
                bytecode.as_deref(),
                &parsed.code,
                parsed.comment.as_deref(),
                index,
                dependency,
            )
        })
        .collect()
}

fn gen_line(
    name: &str,
    expected: ExpectedValue,
    bytecode: Option<&str>,
    code: &str,
    comment: Option<&str>,
    index: usize,
    dependency: Dependency,
) -> String {
    match (expected, comment, dependency) {
        (ExpectedValue::Assert, _, Dependency::Compiler) => {
            format!("\
#[should_panic]
{}
info!(\"test: {{}}\",r##\"{code}\"##);
let src = new_string(r##\"{code}\"##);
let prog = prog(&mut stack,&compiler,src,&runtime,&root,&names,&redef,0.0).expect(\"program compilation failed\");
run(Some(&root),&mut stack,prog.0).unwrap();
}}",
                prefix(name, index, Dependency::Compiler),
            )
        }
        (ExpectedValue::Assert, None, _) => {
            let bytecode = bytecode.unwrap();
            format!(
                "\
#[should_panic]
{}
{{
    info!(\"test: {{}}\",r##\"{code}\"##);
    run(Some(&root),&mut stack,{bytecode}).unwrap()
}};
}}",
                prefix(name, index, dependency)
            )
        }
        (ExpectedValue::Assert, Some(comment), _) => {
            let bytecode = bytecode.unwrap();
            format!(
                "\
#[should_panic]
{}
{{
    info!(\"test: {{}}\",r##\"{comment}\"##);
    run(Some(&root),&mut stack,{bytecode}).unwrap()
}}; // {code}
}}",
                prefix(name, index, dependency)
            )
        }
        (ExpectedValue::Number(expected), _, Dependency::Compiler) => {
            format!("\
{}
info!(\"test: {{}}\",r##\"{code}\"##);
let src = new_string(r##\"{code}\"##);
let prog = prog(&mut stack,&compiler,src,&runtime,&root,&names,&redef,0.0).expect(\"program compilation failed\");
let exec = run(Some(&root),&mut stack,prog.0).unwrap();
assert_eq!(new_scalar({expected}),exec);
}}",
            prefix(name,index,Dependency::Compiler))
        }
        (ExpectedValue::Number(expected), None, dependency) => {
            let bytecode = bytecode.unwrap();
            format!(
                "\
{}
{{
    info!(\"test: {{}}\",r##\"{code}\"##);
    assert_eq!(new_scalar({expected}),run(Some(&root),&mut stack,{bytecode}).unwrap());
}}
}}",
                prefix(name, index, dependency)
            )
        }
        (ExpectedValue::Number(expected), Some(comment), dependency) => {
            let bytecode = bytecode.unwrap();
            format!(
                "\
{}
{{
    info!(\"test: {{}}\",r##\"{comment}\"##);
    assert_eq!(new_scalar({expected}),run(Some(&root),&mut stack,{bytecode}).unwrap());
}} // {code}
}}",
                prefix(name, index, dependency)
            )
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dependency {
    Undefined,
    Compiler,
    Runtime,
}

fn prefix(name: &str, index: usize, dependency: Dependency) -> String {
    use std::fmt::Write;

    match dependency {
        Dependency::Undefined | Dependency::Runtime => {
            let mut generated = format!(
                "\
#[test]
pub fn {name}_{index}() {{
    init_log();
    let mut stack = Stack::new();
    let root = Env::new_root();\n"
            );
            if matches!(dependency, Dependency::Runtime) {
                write!(
                    generated,
                    "    let runtime = runtime(Some(&root),&mut stack).expect(\"runtime failed\")\
                    .into_a().unwrap();"
                )
                .unwrap();
            }
            generated
        }
        Dependency::Compiler => {
            format!(
                "\
#[test]
pub fn {name}_{index}() {{
    init_log();
    let root = Env::new_root();
    let mut stack = Stack::new();
    let runtime = runtime(Some(&root),&mut stack).expect(\"runtime failed\");
    let compiler = run(Some(&root),&mut stack,c(&runtime)).expect(\"compiler failed\");
    let names = V::A(Cc::new(A::new(vec![],vec![0],None)));
    let redef = V::A(Cc::new(A::new(vec![],vec![0],None)));\n"
            )
        }
    }
}
