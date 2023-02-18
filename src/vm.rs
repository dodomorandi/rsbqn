use crate::fmt::{dbg_stack_in, dbg_stack_out};
use crate::runtime::{f, r0, r1};

use crate::coz_scope;
use crate::provide::{decompose, fmtnum, glyph, prim_ind, provide, typ};
use crate::schema::{
    new_scalar, Ar, Block, BlockInst, Bodies, Bytecode, Calleable, Code, Env, Fn, Stack, Stacker,
    Tr2, Tr3, Ve, Vn, Vs, A, D1, D2, V,
};
use bacon_rajan_cc::Cc;

//use std::panic;
use itertools::Itertools;
#[cfg(feature = "debug-ops")]
use log::debug;
use log::info;
use num_traits::FromPrimitive;

pub fn call(stack: &mut Stack, arity: usize, a: Vn, x: Vn, w: Vn) -> Result<Vs, Ve> {
    match a.0 {
        Some(v) => v.call(stack, arity, x, w),
        _ => panic!("unimplemented call"),
    }
}
fn call1(stack: &mut Stack, m: V, f: V) -> Result<Vs, Ve> {
    match m {
        V::BlockInst(ref bl, _prim) => {
            assert_eq!(1, bl.def.typ);
            bl.call_md1(stack, 1, D1::new(m.clone(), f))
        }
        V::R1(_, _prim) => Ok(Vs::V(V::D1(Cc::new(D1::new(m, f)), None))),
        _ => panic!("call1 with invalid type"),
    }
}
fn call2(stack: &mut Stack, m: V, f: V, g: V) -> Result<Vs, Ve> {
    match m {
        V::BlockInst(ref bl, _prim) => {
            assert_eq!(2, bl.def.typ);
            bl.call_md2(stack, 2, D2::new(m.clone(), f, g))
        }
        V::R2(_, _prim) => Ok(Vs::V(V::D2(Cc::new(D2::new(m, f, g)), None))),
        _ => panic!("call2 with invalid type"),
    }
}

fn derv(env: &Env, code: &Cc<Code>, block: &Cc<Block>, stack: &mut Stack) -> Result<Vs, Ve> {
    match (block.typ, block.imm) {
        (0, true) => {
            let child = Env::new(Some(env), block, 0, None);
            let (pos, bodies, body_id) = match &block.bodies {
                Bodies::Comp(b) => {
                    let (p, _l) = code.body_ids[*b];
                    (p, None, None)
                }
                Bodies::Head(amb) => {
                    let (p, _l) = code.body_ids[amb[0]];
                    (p, Some(amb), Some(0))
                }
                _ => panic!("body immediacy derivation doesnt match block definition"),
            };
            vm(&child, code, bodies, body_id, pos, stack)
        }
        (_typ, _imm) => {
            let block_inst = BlockInst::new(env.clone(), block.clone());
            let r = Vs::V(V::BlockInst(Cc::new(block_inst), None));
            Ok(r)
        }
    }
}

fn list(ravel: Vec<V>, fill: Option<V>) -> Vs {
    let shape = vec![ravel.len()];
    Vs::V(V::A(Cc::new(A::new(ravel, shape, fill))))
}
// determines list fill
// use-case for dependent types
fn llst(ravel: Vec<V>) -> Vs {
    let fill = match ravel.iter().all(|e| matches!(e, V::Nothing)) {
        true => Some(new_scalar(0)),
        false => None,
    };
    list(ravel, fill)
}

#[inline]
fn incr(_stack: &mut Stack) {
    #[cfg(feature = "debug-ops")]
    {
        _stack.fp = _stack.s.len();
    }
}

pub fn vm(
    env: &Env,
    code: &Cc<Code>,
    bodies: Option<&Vec<usize>>,
    body_id: Option<usize>,
    pos: usize,
    stack: &mut Stack,
) -> Result<Vs, Ve> {
    incr(stack);
    #[cfg(feature = "debug-ops")]
    debug!("new eval");
    let mut bytecodes_iter = code.get_bc(pos..).bytecodes();
    loop {
        let (pos, bytecode) = bytecodes_iter
            .next()
            .expect("unexpected end of bytecodes")
            .expect("cannot parse bytecode");

        match bytecode {
            Bytecode::Push(x, r) => {
                // PUSH
                coz_scope!("PUSH", {
                    dbg_stack_in("PUSH", pos, format_args!("{} {}", &x, &r), stack);
                    stack.s.push(Vs::V(r.clone()));
                    dbg_stack_out("PUSH", pos, stack);
                });
            }
            Bytecode::Dfnd(x, block) => {
                // DFND
                coz_scope!("DFND", {
                    dbg_stack_in("DFND", pos, format_args!("{}", &x), stack);
                    let r = derv(env, code, block, stack)?;
                    stack.s.push(r);
                    dbg_stack_out("DFND", pos, stack);
                });
            }
            Bytecode::Pops => {
                // POPS
                coz_scope!("POPS", {
                    dbg_stack_in("POPS", pos, "", stack);
                    let _ = stack.s.pop().unwrap();
                    dbg_stack_out("POPS", pos, stack);
                });
            }
            Bytecode::Retn => {
                // RETN
                {}
                coz_scope!("RETN", {
                    dbg_stack_in("RETN", pos, "", stack);
                });
                return Ok(stack.s.pop().unwrap());
            }
            Bytecode::Lsto(x) => {
                // LSTO
                coz_scope!("LSTO", {
                    dbg_stack_in("LSTO", pos, format_args!("{}", &x), stack);
                    let v = stack.s.pop_list(x);
                    stack.s.push(llst(v));
                    dbg_stack_out("LSTO", pos, stack);
                });
            }
            Bytecode::Lstm(x) => {
                // LSTM
                coz_scope!("LSTM", {
                    dbg_stack_in("LSTM", pos, format_args!("{}", &x), stack);
                    let v = stack.s.pop_ref_list(x);
                    stack.s.push(Vs::Ar(Ar::new(v)));
                    dbg_stack_out("LSTM", pos, stack);
                });
            }
            Bytecode::Fn1c => {
                // FN1C
                coz_scope!("FN1C", {
                    dbg_stack_in("FN1C", pos, "", stack);
                    let f = stack.s.pop().unwrap();
                    let x = stack.s.pop().unwrap();
                    let r = call(
                        stack,
                        1,
                        Vn(Some(&f.into_v().unwrap())),
                        Vn(Some(&x.into_v().unwrap())),
                        Vn(None),
                    )?;
                    stack.s.push(r);
                    dbg_stack_out("FN1C", pos, stack);
                });
            }
            Bytecode::Fn1o => {
                // FN1O
                coz_scope!("FN1O", {
                    dbg_stack_in("FN1O", pos, "", stack);
                    let f = stack.s.pop().unwrap();
                    let x = stack.s.pop().unwrap();
                    let r = match &x.as_v().unwrap() {
                        V::Nothing => x,
                        _ => call(
                            stack,
                            1,
                            Vn(Some(&f.into_v().unwrap())),
                            Vn(Some(&x.into_v().unwrap())),
                            Vn(None),
                        )?,
                    };
                    stack.s.push(r);
                    dbg_stack_out("FN1O", pos, stack);
                });
            }
            Bytecode::Fn2c => {
                // FN2C
                coz_scope!("FN2C", {
                    dbg_stack_in("FN2C", pos, "", stack);
                    let w = stack.s.pop().unwrap();
                    let f = stack.s.pop().unwrap();
                    let x = stack.s.pop().unwrap();
                    let r = call(
                        stack,
                        2,
                        Vn(Some(&f.into_v().unwrap())),
                        Vn(Some(&x.into_v().unwrap())),
                        Vn(Some(&w.into_v().unwrap())),
                    )?;
                    stack.s.push(r);
                    dbg_stack_out("FN2C", pos, stack);
                });
            }
            Bytecode::Fn2o => {
                // FN2O
                coz_scope!("FN2O", {
                    dbg_stack_in("FN2O", pos, "", stack);
                    let w = stack.s.pop().unwrap();
                    let f = stack.s.pop().unwrap();
                    let x = stack.s.pop().unwrap();
                    let r = match (&x.as_v().unwrap(), &w.as_v().unwrap()) {
                        (V::Nothing, _) => x,
                        (_, V::Nothing) => call(
                            stack,
                            1,
                            Vn(Some(&f.into_v().unwrap())),
                            Vn(Some(&x.into_v().unwrap())),
                            Vn(None),
                        )?,
                        _ => call(
                            stack,
                            2,
                            Vn(Some(&f.into_v().unwrap())),
                            Vn(Some(&x.into_v().unwrap())),
                            Vn(Some(&w.into_v().unwrap())),
                        )?,
                    };
                    stack.s.push(r);
                    dbg_stack_out("FN2O", pos, stack);
                });
            }
            Bytecode::Tr2d => {
                // TR2D
                coz_scope!("TR2D", {
                    let g = stack.s.pop().unwrap();
                    let h = stack.s.pop().unwrap();
                    dbg_stack_in("TR2D", pos, format_args!("{} {}", &g, &h), stack);
                    let t = Vs::V(V::Tr2(Cc::new(Tr2::new(g, h)), None));
                    stack.s.push(t);
                    dbg_stack_out("TR2D", pos, stack);
                });
            }
            Bytecode::Tr3d => {
                // TR3D
                coz_scope!("TR3D", {
                    dbg_stack_in("TR3D", pos, "", stack);
                    let f = stack.s.pop().unwrap();
                    let g = stack.s.pop().unwrap();
                    let h = stack.s.pop().unwrap();
                    let t = Vs::V(V::Tr3(Cc::new(Tr3::new(f, g, h)), None));
                    stack.s.push(t);
                    dbg_stack_out("TR3D", pos, stack);
                });
            }
            Bytecode::Tr3o => {
                // TR3O
                coz_scope!("TR3O", {
                    dbg_stack_in("TR3O", pos, "", stack);
                    let f = stack.s.pop().unwrap();
                    let g = stack.s.pop().unwrap();
                    let h = stack.s.pop().unwrap();
                    let t = match &f.as_v().unwrap() {
                        V::Nothing => Vs::V(V::Tr2(Cc::new(Tr2::new(g, h)), None)),
                        _ => Vs::V(V::Tr3(Cc::new(Tr3::new(f, g, h)), None)),
                    };
                    stack.s.push(t);
                    dbg_stack_out("TR3O", pos, stack);
                });
            }
            Bytecode::Md1c => {
                // MD1C
                coz_scope!("MD1C", {
                    dbg_stack_in("MD1C", pos, "", stack);
                    let f = stack.s.pop().unwrap();
                    let m = stack.s.pop().unwrap();
                    let r = call1(stack, m.into_v().unwrap(), f.into_v().unwrap())?;
                    stack.s.push(r);
                    dbg_stack_out("MD1C", pos, stack);
                });
            }
            Bytecode::Md2c => {
                // MD2C
                coz_scope!("MD2C", {
                    dbg_stack_in("MD2C", pos, "", stack);
                    let f = stack.s.pop().unwrap();
                    let m = stack.s.pop().unwrap();
                    let g = stack.s.pop().unwrap();
                    let r = call2(
                        stack,
                        m.into_v().unwrap(),
                        f.into_v().unwrap(),
                        g.into_v().unwrap(),
                    )?;
                    stack.s.push(r);
                    dbg_stack_out("MD2C", pos, stack);
                });
            }
            Bytecode::Varo { x, w } => {
                // VARO
                coz_scope!("VARO", {
                    let t = env.ge(x);
                    dbg_stack_in("VARO", pos, format_args!("{} {}", &x, &w), stack);
                    stack.s.push(Vs::V(t.get(w)));
                    dbg_stack_out("VARO", pos, stack);
                });
            }
            Bytecode::Varu { x, w } => {
                // VARU
                coz_scope!("VARU", {
                    let t = env.ge(x);
                    dbg_stack_in("VARU", pos, format_args!("{} {}", &x, &w), stack);
                    stack.s.push(Vs::V(t.get_drop(w)));
                    dbg_stack_out("VARU", pos, stack);
                });
            }
            Bytecode::Varm { x, w } => {
                // VARM
                coz_scope!("VARM", {
                    let t = env.ge(x);
                    dbg_stack_in("VARM", pos, format_args!("{} {}", &x, &w), stack);
                    stack.s.push(Vs::Slot(t.clone(), w));
                    dbg_stack_out("VARM", pos, stack);
                });
            }
            Bytecode::Pred => {
                // PRED
                coz_scope!("PRED", {
                    dbg_stack_in("PRED", pos, "", stack);
                    let pred = stack.s.pop().unwrap();
                    if let Vs::V(v) = &pred {
                        match &v {
                            V::Scalar(n) if *n == 1.0 => (),
                            V::Scalar(n) if *n == 0.0 => {
                                // move to next body in list
                                match (bodies, body_id) {
                                    (Some(b), Some(id)) => {
                                        let (p, locals) = code.body_ids.get(b[id + 1]).unwrap();
                                        return vm(
                                            &env.reinit(*locals),
                                            code,
                                            Some(b),
                                            Some(id + 1),
                                            *p,
                                            stack,
                                        );
                                    }
                                    _ => panic!("no successive body in PRED"),
                                };
                            }
                            _ => panic!("PRED not 0 or 1"),
                        }
                    }
                    dbg_stack_out("PRED", pos, stack);
                });
            }
            Bytecode::Vfym => {
                // VFYM
                coz_scope!("VFYM", {
                    let m = stack.s.pop().unwrap();
                    dbg_stack_in("VFYM", pos, "", stack);
                    stack.s.push(Vs::Match(Some(m.into_v().unwrap())));
                    dbg_stack_out("VFYM", pos, stack);
                });
            }
            Bytecode::Notm => {
                // NOTM
                coz_scope!("NOTM", {
                    dbg_stack_in("NOTM", pos, "", stack);
                    stack.s.push(Vs::Match(None));
                    dbg_stack_out("NOTM", pos, stack);
                });
            }
            Bytecode::Seth => {
                // SETH
                coz_scope!("SETH", {
                    dbg_stack_in("SETH", pos, "", stack);
                    let i = stack.s.pop().unwrap();
                    let v = stack.s.pop().unwrap();
                    match i.set(true, v.as_v().unwrap()) {
                        Ok(_r) => (), // continue
                        Err(_) => {
                            // move to next body in list
                            match (bodies, body_id) {
                                (Some(b), Some(id)) => {
                                    let (p, locals) = code.body_ids.get(b[id + 1]).unwrap();
                                    return vm(
                                        &env.reinit(*locals),
                                        code,
                                        Some(b),
                                        Some(id + 1),
                                        *p,
                                        stack,
                                    );
                                }
                                _ => panic!("no successive body in SETH"),
                            };
                        }
                    }
                    dbg_stack_out("SETH", pos, stack);
                });
            }
            Bytecode::Setn => {
                // SETN
                coz_scope!("SETN", {
                    dbg_stack_in("SETN", pos, "", stack);
                    let i = stack.s.pop().unwrap();
                    let v = stack.s.pop().unwrap();
                    let r = i.set(true, v.as_v().unwrap())?;
                    stack.s.push(Vs::V(r));
                    dbg_stack_out("SETN", pos, stack);
                });
            }
            Bytecode::Setu => {
                // SETU
                coz_scope!("SETU", {
                    dbg_stack_in("SETU", pos, "", stack);
                    let i = stack.s.pop().unwrap();
                    let v = stack.s.pop().unwrap();
                    let r = i.set(false, v.as_v().unwrap())?;
                    stack.s.push(Vs::V(r));
                    dbg_stack_out("SETU", pos, stack);
                });
            }
            Bytecode::Setm => {
                // SETM
                coz_scope!("SETM", {
                    dbg_stack_in("SETM", pos, "", stack);
                    let i = stack.s.pop().unwrap();
                    let f = stack.s.pop().unwrap();
                    let x = stack.s.pop().unwrap();
                    let v = call(
                        stack,
                        2,
                        Vn(Some(&f.into_v().unwrap())),
                        Vn(Some(&x.into_v().unwrap())),
                        Vn(Some(&i.get())),
                    )?;
                    let r = i.set(false, v.as_v().unwrap())?;
                    stack.s.push(Vs::V(r));
                    dbg_stack_out("SETM", pos, stack);
                });
            }
            Bytecode::Setc => {
                // SETC
                coz_scope!("SETC", {
                    dbg_stack_in("SETC", pos, "", stack);
                    let i = stack.s.pop().unwrap();
                    let f = stack.s.pop().unwrap();
                    let v = call(
                        stack,
                        1,
                        Vn(Some(&f.into_v().unwrap())),
                        Vn(Some(&i.get())),
                        Vn(None),
                    )?;
                    let r = i.set(false, v.as_v().unwrap())?;
                    stack.s.push(Vs::V(r));
                    dbg_stack_out("SETC", pos, stack);
                });
            }
            b => unimplemented!("bytecode {b:?}"),
        }

        #[cfg(feature = "coz-loop")]
        coz::progress!();
    }
}

pub fn runtime(root: Option<&Env>, stack: &mut Stack) -> Result<V, Ve> {
    let builtin = provide();
    let runtime0 = run(root, stack, r0(&builtin))?;
    let runtime1 = run(root, stack, r1(&builtin, &runtime0))?;
    info!("runtime0 loaded");
    match runtime1.into_a().unwrap().get_mut() {
        Some(full_runtime) => {
            let _set_inv = full_runtime.r.pop().unwrap();
            let set_prims = full_runtime.r.pop().unwrap();
            let runtime = full_runtime.r.pop().unwrap();

            // Copy-On-Write. Use two assignments to prevent tmp values freed while in use.
            let mut prims = runtime.into_a().unwrap();
            let mut_prims = prims.make_unique();

            // set primitive indices
            for i in 0..mut_prims.r.len() - 1 {
                let e = &mut mut_prims.r[i];
                match e {
                    V::UserMd1(_b, _a, ref mut prim) => *prim = Some(i),
                    V::UserMd2(_b, _a, ref mut prim) => *prim = Some(i),
                    V::BlockInst(_b, ref mut prim) => *prim = Some(i),
                    V::Fn(_a, ref mut prim) => *prim = Some(i),
                    V::R1(_r1, ref mut prim) => *prim = Some(i),
                    V::R2(_r2, ref mut prim) => *prim = Some(i),
                    V::D1(_d1, ref mut prim) => *prim = Some(i),
                    V::D2(_d2, ref mut prim) => *prim = Some(i),
                    V::Tr2(_tr2, ref mut prim) => *prim = Some(i),
                    V::Tr3(_tr3, ref mut prim) => *prim = Some(i),
                    _ => panic!("illegal setprim"),
                }
            }
            info!("runtime loaded");
            let prim_fns = V::A(Cc::new(A::new(
                vec![V::Fn(Fn(decompose), None), V::Fn(Fn(prim_ind), None)],
                vec![2],
                None,
            )));
            let _ = call(
                stack,
                1,
                Vn(Some(&set_prims)),
                Vn(Some(&prim_fns)),
                Vn(None),
            );
            Ok(V::A(prims))
        }
        None => panic!("cant get mutable runtime"),
    }
}

pub fn sysfns(_arity: usize, _x: Vn, _w: Vn) -> Result<Vs, Ve> {
    Ok(Vs::V(V::A(Cc::new(A::new(vec![], vec![0], None)))))
}

#[allow(clippy::too_many_arguments)]
pub fn prog(
    stack: &mut Stack,
    compiler: &V,
    src: V,
    runtime: &V,
    env: &Env,
    names: &V,
    redef: &V,
    strictness: f64,
) -> Result<(Cc<Code>, A, A), Ve> {
    // an array ravel is a vector of owned values
    // because we are passing the vars/names/redefs as elements in an array, they must be moved to the new prog
    // this will likely result in excess clones
    let vars = env.to_vars();
    let args = V::A(Cc::new(A::new(
        vec![
            runtime.clone(),
            V::Fn(Fn(sysfns), None),
            names.clone(),
            redef.clone(),
        ],
        vec![4],
        None,
    )));
    let mut prog = call(
        stack,
        2,
        Vn(Some(compiler)),
        Vn(Some(&src)),
        Vn(Some(&args)),
    )?
    .into_v()
    .unwrap()
    .into_a()
    .unwrap();
    match prog.get_mut() {
        Some(p) => {
            let tokenization = p.r.pop().unwrap();
            let _indices = p.r.pop().unwrap();
            let bodies = p.r.pop().unwrap();
            let blocks = p.r.pop().unwrap();
            let objects = p.r.pop().unwrap();
            let bytecode = p.r.pop().unwrap();

            // repl stuff
            let varlen = vars.as_a().unwrap().r.len();
            //info!("varlen = {}",&varlen);
            let pnames = &tokenization.as_a().unwrap().r[2].as_a().unwrap().r[0]
                .as_a()
                .unwrap()
                .r;
            //info!("pnames = {:?}",&pnames);
            let newv = &bodies.as_a().unwrap().r[0].as_a().unwrap().r[2]
                .as_a()
                .unwrap()
                .r[varlen..];
            //info!("newv = {:?}",newv);

            let mut namestmp = names.as_a().unwrap().clone();
            let namesmut = namestmp.make_unique();
            let mut newnames = newv
                .iter()
                .map(|i| pnames[usize::from_f64(*i.as_scalar().unwrap()).unwrap()].clone())
                .collect::<Vec<V>>();
            namesmut.r.append(&mut newnames);
            namesmut.sh = vec![namesmut.r.len()];

            let mut redeftmp = redef.as_a().unwrap().clone();
            let redefmut = redeftmp.make_unique();
            let mut newredef = newv
                .iter()
                .map(|_i| V::Scalar(strictness))
                .collect::<Vec<V>>();
            redefmut.r.append(&mut newredef);
            redefmut.sh = vec![redefmut.r.len()];

            // extend root env
            env.extend(newv.len());
            // end repl stuff

            let bc = bytecode
                .as_a()
                .unwrap()
                .r
                .iter()
                .map(|e| match e {
                    V::Scalar(n) => usize::from_f64(*n).unwrap(),
                    _ => panic!("bytecode not a number"),
                })
                .collect();
            let objs = match objects.into_a().unwrap().try_unwrap() {
                Ok(o) => o.r,
                Err(_o) => panic!("objects not unique"),
            };
            let blocks_raw = match blocks.into_a().unwrap().try_unwrap() {
                Ok(b) => b
                    .r
                    .iter()
                    .map(|e| match e.as_a().unwrap().r.iter().collect_tuple() {
                        Some((V::Scalar(typ), V::Scalar(imm), V::Scalar(body))) => (
                            u8::from_f64(*typ).unwrap(),
                            1.0 == *imm,
                            Bodies::Comp(usize::from_f64(*body).unwrap()),
                        ),
                        Some((V::Scalar(typ), V::Scalar(imm), V::A(bodies))) => {
                            match bodies.r.len() {
                                1 => {
                                    let amb = &bodies.r[0];
                                    (
                                        u8::from_f64(*typ).unwrap(),
                                        1.0 == *imm,
                                        Bodies::Head(
                                            amb.as_a()
                                                .unwrap()
                                                .r
                                                .iter()
                                                .map(|e| match e {
                                                    V::Scalar(n) => usize::from_f64(*n).unwrap(),
                                                    _ => panic!("bytecode not a number"),
                                                })
                                                .collect::<Vec<usize>>(),
                                        ),
                                    )
                                }
                                2 => {
                                    let (mon, dya) = bodies.r.iter().collect_tuple().unwrap();
                                    (
                                        u8::from_f64(*typ).unwrap(),
                                        1.0 == *imm,
                                        Bodies::Exp(
                                            mon.as_a()
                                                .unwrap()
                                                .r
                                                .iter()
                                                .map(|e| match e {
                                                    V::Scalar(n) => usize::from_f64(*n).unwrap(),
                                                    _ => panic!("bytecode not a number"),
                                                })
                                                .collect::<Vec<usize>>(),
                                            dya.as_a()
                                                .unwrap()
                                                .r
                                                .iter()
                                                .map(|e| match e {
                                                    V::Scalar(n) => usize::from_f64(*n).unwrap(),
                                                    _ => panic!("bytecode not a number"),
                                                })
                                                .collect::<Vec<usize>>(),
                                        ),
                                    )
                                }
                                _ => panic!("unaccounted for headers len"),
                            }
                        }
                        _ => panic!("couldn't load compiled block"),
                    })
                    .collect::<Vec<(u8, bool, Bodies)>>(),
                Err(_b) => panic!("cant get unique ref to program blocks"),
            };
            let bods = match bodies.into_a().unwrap().try_unwrap() {
                Ok(b) => {
                    b.r.iter()
                        .map(|e| match e.as_a().unwrap().r.iter().collect_tuple() {
                            Some((V::Scalar(pos), V::Scalar(local), _name_id, _export_mask)) => (
                                usize::from_f64(*pos).unwrap(),
                                usize::from_f64(*local).unwrap(),
                            ),
                            _x => panic!("couldn't load compiled body"),
                        })
                        .collect::<Vec<(usize, usize)>>()
                }
                Err(_b) => panic!("cant get unique ref to program blocks"),
            };
            Ok((
                Code::new(bc, objs, blocks_raw, bods),
                namesmut.to_owned(),
                redefmut.to_owned(),
            ))
        }
        None => panic!("cant get unique ref to blocks"),
    }
}

pub fn formatter(root: Option<&Env>, stack: &mut Stack, runtime: &V) -> Result<V, Ve> {
    let formatter = run(root, stack, f(runtime)).expect("couldnt load fmt");
    let fmt_fns = V::A(Cc::new(A::new(
        vec![
            V::Fn(Fn(typ), None),
            V::Fn(Fn(decompose), None),
            V::Fn(Fn(glyph), None),
            V::Fn(Fn(fmtnum), None),
        ],
        vec![4],
        None,
    )));
    let mut fmt = call(stack, 1, Vn(Some(&formatter)), Vn(Some(&fmt_fns)), Vn(None))
        .expect("fmt malformed")
        .into_v()
        .unwrap()
        .into_a()
        .unwrap();
    let fmt1 = fmt.make_unique();
    let _repr = fmt1.r.pop().unwrap();
    let fmt1 = fmt1.r.pop().unwrap();
    Ok(fmt1)
}

pub fn run_in_place(env: &Env, stack: &mut Stack, code: Cc<Code>) -> Result<V, Ve> {
    let pos = match code.blocks[0].bodies {
        Bodies::Comp(b) => {
            let (pos, _locals) = code.body_ids[b];
            pos
        }
        Bodies::Head(_) => panic!("cant run Head bodies"),
        Bodies::Exp(_, _) => panic!("cant run Expanded bodies"),
    };
    match vm(env, &code, None, None, pos, stack) {
        Ok(r) => Ok(r.into_v().unwrap()),
        Err(e) => Err(e),
    }
}
pub fn run(parent: Option<&Env>, stack: &mut Stack, code: Cc<Code>) -> Result<V, Ve> {
    let child = Env::new(parent, &code.blocks[0], 0, None);
    run_in_place(&child, stack, code)
}
