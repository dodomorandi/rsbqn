use bacon_rajan_cc::Cc;
use log::info;
use rsbqn::code::c;
use rsbqn::fmt::{fmt_err, fmt_result};
use rsbqn::init_log;
use rsbqn::schema::{new_string, Env, Stack, Ve, Vn, A, V};
use rsbqn::vm::{call, formatter, prog, run, run_in_place, runtime};
use rustyline::error::ReadlineError;
use rustyline::{Editor, Result};

#[cfg(feature = "dhat")]
use dhat::{Dhat, DhatAlloc};

#[cfg(feature = "dhat")]
#[global_allocator]
static ALLOCATOR: DhatAlloc = DhatAlloc;

fn main() -> Result<()> {
    init_log();
    #[cfg(feature = "dhat")]
    let _dhat = Dhat::start_heap_profiling();

    let mut stack = Stack::new();
    let root = Env::new_root();

    let runtime = runtime(Some(&root), &mut stack).expect("couldnt load runtime");
    let compiler = run(Some(&root), &mut stack, c(&runtime)).expect("couldnt load compiler");
    let fmt = formatter(Some(&root), &mut stack, &runtime).expect("couldnt load formatter");
    // initialize names/redef to empty arrays of size 0
    let mut names = V::A(Cc::new(A::new(vec![], vec![0], None)));
    let mut redef = V::A(Cc::new(A::new(vec![], vec![0], None)));

    let mut rl = Editor::<()>::new();
    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                rl.add_history_entry(line.as_str());
                let src = new_string(&line);
                match prog(
                    &mut stack, &compiler, src, &runtime, &root, &names, &redef, -1.0,
                ) {
                    Ok((prog, newnames, newredef)) => {
                        names = V::A(Cc::new(newnames));
                        redef = V::A(Cc::new(newredef));
                        info!("names = {}", &names);
                        info!("redef = {}", &redef);
                        match run_in_place(&root, &mut stack, prog) {
                            Ok(exec) => {
                                match call(&mut stack, 1, Vn(Some(&fmt)), Vn(Some(&exec)), Vn(None))
                                {
                                    Ok(f) => {
                                        println!(
                                            "{}",
                                            fmt_result(&f.into_v().unwrap().into_a().unwrap())
                                        )
                                    }
                                    Err(e) => println!("{}", e),
                                }
                            }
                            Err(e) => match e {
                                Ve::S(s) => println!("{}", s),
                                Ve::V(v) => match v {
                                    V::A(a) => println!("{}", fmt_err(&a.r)),
                                    V::Scalar(n) => println!("Err: {}", n),
                                    _ => panic!("can't error on type"),
                                },
                            },
                        };
                    }
                    Err(e) => match e {
                        Ve::S(s) => println!("{}", s),
                        Ve::V(v) => match v {
                            V::A(a) => match a.r.len() {
                                2 => println!("{}", fmt_err(&a.r[1].as_a().unwrap().r)),
                                _ => println!("{}", fmt_err(&a.r)),
                            },
                            _ => panic!("cant error on type"),
                        },
                    },
                };
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
    // single line variations for copy-pasting
    //{ let result = call(1,Some(&run(prog(&compiler,new_string("{×´1+↕𝕩}"),&runtime))),Some(&V::Scalar(10.0)),None); println!("{}",result); }
    //{ let runtimev = runtime(); let runtime = runtimev.as_a().unwrap();let compiler = c(&runtimev); }
    //{ let runtimev = runtime(); }
    //{ let runtimev = runtime(); let runtime = runtimev.as_a().unwrap();let compiler = c(&runtimev); call(1,Some(&run(prog(&compiler,src,&runtime))),Some(&V::Scalar(10.0)),None); }
    //{ let runtimev = runtime(); let runtime = runtimev.as_a().unwrap();let compiler = c(&runtimev); let result = call(1,Some(&run(prog(&compiler,src,&runtime))),Some(&V::Scalar(10.0)),None); println!("{}",result); }
}
