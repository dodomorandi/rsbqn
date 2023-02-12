use crate::schema::{
    new_char, new_scalar, new_string, Decoder, Fn, Stack, Tr2, Tr3, Ve, Vn, Vs, A, D1, D2, R1, R2,
    V,
};
use crate::vm::call;
use bacon_rajan_cc::Cc;

#[cfg(feature = "debug-fns")]
use log::debug;
use num_traits::cast::FromPrimitive;
use std::char;

use std::ops::Deref;
use std::ops::Mul;

#[cfg(feature = "debug-fns")]
fn dbg_args(fun: &str, arity: usize, x: &Vn, w: &Vn) {
    match arity {
        1 => {
            debug!(
                "calling {}/{}: 𝕩 = {}",
                fun,
                arity,
                format!("{}", x.0.unwrap())
            );
        }
        2 => {
            debug!(
                "calling {}/{}: 𝕩 = {};𝕨 = {}",
                fun,
                arity,
                format!("{}", x.0.unwrap()),
                format!("{}", w.0.unwrap())
            );
        }
        _ => (),
    };
}

#[cfg(feature = "debug-fns")]
fn dbg_rtn(fun: &str, arity: usize, r: &Result<Vs, Ve>) {
    match r {
        Ok(r) => debug!("rtn     {}/{}: rtn = {}", fun, arity, r),
        Err(e) => debug!("rtn     {}/{}: rtn = {}", fun, arity, e),
    }
}

// Type
pub fn typ(arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("typ");
    match arity {
        1 => {
            let out = match x.0.unwrap() {
                V::Scalar(_n) => 1.0,
                V::A(_a) => 0.0,
                V::Char(_c) => 2.0,
                V::UserMd1(_b, _a, _prim) => 3.0,
                V::UserMd2(_b, _a, _prim) => 3.0,
                V::D1(_d1, _prim) => 3.0,
                V::D2(_d2, _prim) => 3.0,
                V::Tr2(_tr3, _prim) => 3.0,
                V::Tr3(_tr3, _prim) => 3.0,
                V::Fn(_fn, _prim) => 3.0,
                V::R1(_r1, _prim) => 4.0,
                V::R2(_r2, _prim) => 5.0,
                V::BlockInst(b, _prim) => b.def.typ as f64 + 3.0,
                V::Nothing => return Err(Ve::S("no matching value for typ")),
            };

            Ok(Vs::V(V::Scalar(out)))
        }
        _ => panic!("typ arity not implemented"),
    }
}
// Fill
fn fill(arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("fill");
    match arity {
        1 => Ok(Vs::V(V::Scalar(0.0))),
        2 => Ok(Vs::V(x.0.unwrap().clone())),
        _ => panic!("illegal fill arity"),
    }
}
// Log
fn log(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("log");
    match arity {
        1 => match x.0.unwrap() {
            V::Scalar(xs) => Ok(Vs::V(V::Scalar(xs.ln()))),
            _ => Err(Ve::S("monadic log expected number")),
        },
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::Scalar(xs), V::Scalar(ws)) => Ok(Vs::V(V::Scalar(xs.ln() / ws.ln()))),
            _ => Err(Ve::S("dyadic log expected numbers")),
        },
        _ => panic!("illegal power arity"),
    }
}
// GroupLen
fn group_len(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("group_len");
    match arity {
        1 => match x.0.unwrap() {
            V::A(xa) => {
                let l = xa.r.iter().fold(-1.0, |acc, i| i.to_f64().max(acc));
                let s = l + 1.0;
                let mut r = vec![V::Scalar(0.0); s as usize];
                let mut i = 0;
                while i < xa.r.len() {
                    let e = xa.r[i].to_f64();
                    if e >= 0.0 {
                        r[e as usize] = V::Scalar(r[e as usize].to_f64() + 1.0)
                    }
                    i += 1;
                }
                Ok(Vs::V(V::A(Cc::new(A::new(
                    r.clone(),
                    vec![r.len()],
                    Some(new_scalar(0)),
                )))))
            }
            _ => Err(Ve::S("group_len 𝕩 is not an array")),
        },
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::A(xa), V::Scalar(ws)) => {
                let l = xa.r.iter().fold(ws - 1.0, |acc, i| i.to_f64().max(acc));
                let s = l + 1.0;
                let mut r = vec![V::Scalar(0.0); s as usize];
                let mut i = 0;
                while i < xa.r.len() {
                    let e = xa.r[i].to_f64();
                    if e >= 0.0 {
                        r[e as usize] = V::Scalar(r[e as usize].to_f64() + 1.0)
                    }
                    i += 1;
                }
                Ok(Vs::V(V::A(Cc::new(A::new(
                    r.clone(),
                    vec![r.len()],
                    Some(new_scalar(0)),
                )))))
            }
            _ => Err(Ve::S("group_len 𝕩 is not an array")),
        },
        _ => panic!("illegal group_len arity"),
    }
}
// GroupOrd
fn group_ord(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("group_ord");
    match arity {
        2 => match (&x.0.unwrap(), &w.0.unwrap()) {
            (V::A(xa), V::A(wa)) => {
                let (mut s, l) = wa.r.iter().fold((vec![], 0.0), |(mut si, li), v| {
                    si.push(li);
                    (si, li + v.to_f64())
                });
                let mut r = vec![V::Nothing; l as usize];
                xa.r.iter().enumerate().for_each(|(i, e)| {
                    if e.to_f64() >= 0.0 {
                        r[s[e.to_f64() as usize] as usize] = V::Scalar(i as f64);
                        s[e.to_f64() as usize] += 1.0;
                    }
                });
                let shape = vec![r.len()];
                Ok(Vs::V(V::A(Cc::new(A::new(r, shape, None)))))
            }
            _ => Err(Ve::S("dyadic group_ord x is not an array")),
        },
        _ => panic!("illegal group_ord arity"),
    }
}
// !
fn assert_fn(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "debug-fns")]
    dbg_args("assert_fn", arity, &x, &w);
    let r = match arity {
        1 => match x.0.unwrap().as_scalar() {
            Some(n) if *n == 1.0 => Ok(Vs::V(V::Scalar(1.0))),
            _ => Err(Ve::V(x.0.unwrap().clone())),
        },
        2 => match x.0.unwrap().as_scalar() {
            Some(n) if *n == 1.0 => Ok(Vs::V(V::Scalar(1.0))),
            _ => {
                //let msg = w.0.unwrap().as_a().unwrap().r.iter().map(|e| match e {
                //    V::Char(c) => *c,
                //    _ => panic!("panic is not a string"),
                //}).collect::<Vec<char>>();
                //Err(String::from_iter(&msg).to_owned())
                Err(Ve::V(w.0.unwrap().clone()))
            }
        },
        _ => panic!("illegal assert arity"),
    };
    #[cfg(feature = "debug-fns")]
    dbg_rtn("assert_fn", arity, &r);
    r
}
// +
pub fn plus(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("plus");
    #[cfg(feature = "debug-fns")]
    dbg_args("plus", arity, &x, &w);

    match arity {
        1 => Ok(Vs::V({ x.0.unwrap() }.clone())),
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::Char(xc), V::Scalar(ws)) if *ws >= 0.0 => Ok(Vs::V(V::Char(
                char::from_u32(u32::from(*xc) + u32::from_f64(*ws).unwrap()).unwrap(),
            ))),
            (V::Scalar(xs), V::Char(wc)) if *xs >= 0.0 => Ok(Vs::V(V::Char(
                char::from_u32(u32::from(*wc) + u32::from_f64(*xs).unwrap()).unwrap(),
            ))),
            (V::Char(xc), V::Scalar(ws)) if *ws < 0.0 => Ok(Vs::V(V::Char(
                char::from_u32(u32::from(*xc) - u32::from_f64(ws.abs()).unwrap()).unwrap(),
            ))),
            (V::Scalar(xs), V::Char(wc)) if *xs < 0.0 => Ok(Vs::V(V::Char(
                char::from_u32(u32::from(*wc) - u32::from_f64(xs.abs()).unwrap()).unwrap(),
            ))),
            (V::Scalar(xs), V::Scalar(ws)) => Ok(Vs::V(V::Scalar(xs + ws))),
            _ => Err(Ve::S("dyadic plus pattern not found")),
        },
        _ => panic!("illegal plus arity"),
    }
}
// -
fn minus(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("minus");
    #[cfg(feature = "debug-fns")]
    dbg_args("minus", arity, &x, &w);

    match arity {
        1 => match x.0.unwrap() {
            V::Scalar(xs) => Ok(Vs::V(V::Scalar(-1.0 * xs))),
            _ => Err(Ve::S("monadic minus expected number")),
        },
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::Scalar(xs), V::Char(wc)) => Ok(Vs::V(V::Char(
                char::from_u32(u32::from(*wc) - u32::from_f64(*xs).unwrap()).unwrap(),
            ))),
            (V::Char(xc), V::Char(wc)) if u32::from(*xc) > u32::from(*wc) => Ok(Vs::V(V::Scalar(
                -1.0 * f64::from(u32::from(*xc) - u32::from(*wc)),
            ))),
            (V::Char(xc), V::Char(wc)) => {
                Ok(Vs::V(V::Scalar(f64::from(u32::from(*wc) - u32::from(*xc)))))
            }
            (V::Scalar(xs), V::Scalar(ws)) => Ok(Vs::V(V::Scalar(ws - xs))),
            _ => Err(Ve::S("dyadic minus pattern not found")),
        },
        _ => panic!("illegal minus arity"),
    }
}
// ×
fn times(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("times");
    #[cfg(feature = "debug-fns")]
    dbg_args("times", arity, &x, &w);
    match arity {
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::Scalar(xs), V::Scalar(ws)) => Ok(Vs::V(V::Scalar(ws * xs))),
            _ => Err(Ve::S("dyadic times illegal arguments")),
        },
        _ => panic!("illegal times arity"),
    }
}
// ÷
fn divide(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("divide");
    #[cfg(feature = "debug-fns")]
    dbg_args("divide", arity, &x, &w);
    match arity {
        1 => match x.0.unwrap() {
            V::Scalar(xs) => Ok(Vs::V(V::Scalar(1.0 / xs))),
            _ => Err(Ve::S("monadic divide expected number")),
        },
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::Scalar(xs), V::Scalar(ws)) => Ok(Vs::V(V::Scalar(ws / xs))),
            _ => Err(Ve::S("dyadic divide expected number")),
        },
        _ => panic!("illegal divide arity"),
    }
}
// ⋆
fn power(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("power");
    match arity {
        1 => match x.0.unwrap() {
            V::Scalar(xs) => Ok(Vs::V(V::Scalar(xs.exp()))),
            _ => Err(Ve::S("monadic power expected number")),
        },
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::Scalar(xs), V::Scalar(ws)) => Ok(Vs::V(V::Scalar(ws.powf(*xs)))),
            _ => Err(Ve::S("dyadic power expected numbers")),
        },
        _ => panic!("illegal power arity"),
    }
}
// ⌊
fn floor(arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("floor");
    match arity {
        1 | 2 => Ok(Vs::V(V::Scalar(x.0.unwrap().to_f64().floor()))),
        _ => panic!("illegal divide arity"),
    }
}
// =
fn equals(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("equals");
    #[cfg(feature = "debug-fns")]
    dbg_args("equals", arity, &x, &w);

    match arity {
        1 => match x.0.unwrap() {
            V::A(xa) => Ok(Vs::V(V::Scalar(xa.sh.len() as i64 as f64))),
            V::Char(_xc) => Ok(Vs::V(V::Scalar(0.0))),
            V::Scalar(_xs) => Ok(Vs::V(V::Scalar(0.0))),
            V::UserMd1(_b, _a, _prim) => Ok(Vs::V(V::Scalar(0.0))),
            V::UserMd2(_b, _a, _prim) => Ok(Vs::V(V::Scalar(0.0))),
            V::D2(_d2, _prim) => Ok(Vs::V(V::Scalar(0.0))),
            V::BlockInst(_b, _prim) => Ok(Vs::V(V::Scalar(0.0))),
            V::Nothing => panic!("monadic equals nothing invalid"),
            V::Fn(_a, _prim) => panic!("monadic equals fn invalid"),
            V::R1(_f, _prim) => panic!("monadic equals r1 invalid"),
            V::R2(_f, _prim) => panic!("monadic equals r2 invalid"),
            V::D1(_d1, _prim) => panic!("monadic equals d1 invalid"),
            V::Tr2(_tr2, _prim) => panic!("monadic equals train2 invalid"),
            V::Tr3(_tr3, _prim) => panic!("monadic equals train3 invalid"),
        },
        2 => match x.0.unwrap() == w.0.unwrap() {
            true => Ok(Vs::V(V::Scalar(1.0))),
            false => Ok(Vs::V(V::Scalar(0.0))),
        },
        _ => panic!("illegal equals arity"),
    }
}
// ≤
fn lesseq(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("lesseq");
    #[cfg(feature = "debug-fns")]
    dbg_args("lesseq", arity, &x, &w);
    let r = match arity {
        2 => {
            let t = typ(1, Vn(x.0), Vn(None))?.as_v().unwrap().to_f64();
            let s = typ(1, Vn(w.0), Vn(None))?.as_v().unwrap().to_f64();
            if x.0.as_ref().unwrap().is_fn() || w.0.as_ref().unwrap().is_fn() {
                Err(Ve::S("cannot compare operations"))
            } else {
                match t != s {
                    true => Ok(Vs::V(V::Scalar((s <= t) as i64 as f64))),
                    false => Ok(Vs::V(V::Scalar(
                        (w.0.unwrap().to_f64() <= x.0.unwrap().to_f64()) as i64 as f64,
                    ))),
                }
            }
        }
        _ => panic!("illegal lesseq arity"),
    };
    #[cfg(feature = "debug-fns")]
    dbg_rtn("lesseq", arity, &r);
    r
}
// ≢
fn shape(arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("shape");
    match arity {
        1 => match x.0.unwrap() {
            V::A(xa) => {
                let ravel = xa
                    .sh
                    .iter()
                    .map(|n| V::Scalar(*n as i64 as f64))
                    .collect::<Vec<V>>();
                let shape = vec![ravel.len()];
                Ok(Vs::V(V::A(Cc::new(A::new(
                    ravel,
                    shape,
                    Some(new_scalar(0)),
                )))))
            }
            _ => Err(Ve::S("shape 𝕩 is not an array")),
        },
        _ => panic!("illegal shape arity"),
    }
}
// ⥊
fn reshape(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("reshape");
    match arity {
        1 => match x.0.unwrap() {
            V::A(xa) => Ok(Vs::V(V::A(Cc::new(A::new(
                xa.r.clone(),
                vec![xa.r.len()],
                xa.fill.clone(),
            ))))),
            _ => Err(Ve::S("monadic reshape no arr")),
        },
        2 => match ({ x.0.unwrap() }, w.0.unwrap()) {
            (V::A(ax), V::A(aw)) => {
                let sh =
                    aw.r.iter()
                        .map(|e| match e {
                            V::Scalar(n) => *n as usize,
                            _ => panic!("W ravel is not a num"),
                        })
                        .collect::<Vec<usize>>();
                Ok(Vs::V(V::A(Cc::new(A::new(
                    ax.r.clone(),
                    sh,
                    ax.fill.clone(),
                )))))
            }
            _ => Err(Ve::S("dydic reshape no match")),
        },
        _ => panic!("illegal reshape arity"),
    }
}
// ⊑
fn pick(arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("pick");
    #[cfg(feature = "debug-fns")]
    dbg_args("pick", arity, &x, &w);

    match arity {
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::A(a), V::Scalar(i)) if *i >= 0.0 => Ok(Vs::V(a.r[*i as i64 as usize].clone())),
            (V::A(a), V::Scalar(i)) if *i < 0.0 => {
                Ok(Vs::V(a.r[((a.r.len() as f64) + i) as i64 as usize].clone()))
            }
            _ => Err(Ve::S("pick - can't index into non array")),
        },
        _ => panic!("illegal pick arity"),
    }
}
// ↕
fn windows(arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("windows");
    match arity {
        1 => match x.0.unwrap() {
            V::Scalar(n) => Ok(Vs::V(V::A(Cc::new(A::new(
                (0..*n as i64)
                    .map(|v| V::Scalar(v as f64))
                    .collect::<Vec<V>>(),
                vec![*n as usize],
                Some(new_scalar(0)),
            ))))),
            _ => Err(Ve::S("x is not a number")),
        },
        _ => panic!("illegal windows arity"),
    }
}
// ⌜
fn table(stack: &mut Stack, arity: usize, f: Vn, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("table");
    match arity {
        1 => match x.0.unwrap() {
            V::A(xa) => {
                let mut ravel: Vec<V> = Vec::with_capacity(xa.r.len());
                for i in 0..xa.r.len() {
                    ravel.push(
                        match call(stack, arity, Vn(f.0), Vn(Some(&xa.r[i])), Vn(None)) {
                            Ok(r) => r.into_v().unwrap(),
                            Err(e) => return Err(e),
                        },
                    )
                }
                let sh = xa.sh.clone();
                Ok(Vs::V(V::A(Cc::new(A::new(ravel, sh, None)))))
            }
            _ => Err(Ve::S("monadic table x is not an array")),
        },
        2 => match (x.0.unwrap(), w.0.unwrap()) {
            (V::A(xa), V::A(wa)) => {
                let mut ravel: Vec<V> = Vec::with_capacity(xa.r.len().mul(wa.r.len()));
                for i in 0..wa.r.len() {
                    for j in 0..xa.r.len() {
                        ravel.push(
                            match call(
                                stack,
                                arity,
                                Vn(f.0),
                                Vn(Some(&xa.r[j])),
                                Vn(Some(&wa.r[i])),
                            ) {
                                Ok(r) => r.into_v().unwrap(),
                                Err(e) => return Err(e),
                            },
                        )
                    }
                }
                let sh = wa
                    .sh
                    .clone()
                    .into_iter()
                    .chain(xa.sh.clone().into_iter())
                    .collect();
                Ok(Vs::V(V::A(Cc::new(A::new(ravel, sh, None)))))
            }
            _ => Err(Ve::S("dyadic table not an array")),
        },
        _ => panic!("illegal table arity"),
    }
}
// `
fn scan(stack: &mut Stack, arity: usize, f: Vn, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("scan");
    match arity {
        1 => match x.0.unwrap() {
            V::A(a) => {
                let s = &a.sh;
                if s.is_empty() {
                    return Err(Ve::S("scan monadic array rank not at least 1"));
                };
                let l = a.r.len();
                let mut r = vec![V::Nothing; l];
                if l > 0 {
                    let mut c = 1;
                    let mut i = 1;
                    while i < s.len() {
                        c *= s[i];
                        i += 1;
                    }
                    i = 0;
                    while i < c {
                        r[i] = a.r[i].clone();
                        i += 1;
                    }
                    while i < l {
                        r[i] = match call(stack, 2, Vn(f.0), Vn(Some(&a.r[i])), Vn(Some(&r[i - c])))
                        {
                            Ok(v) => v.as_v().unwrap().clone(),
                            Err(_e) => panic!("monadic scan call failed"),
                        };
                        i += 1;
                    }
                };
                Ok(Vs::V(V::A(Cc::new(A::new(r, s.to_vec(), a.fill.clone())))))
            }
            _ => Err(Ve::S("monadic scan x is not an array")),
        },
        2 => {
            let (wr, wa) = match w.0.unwrap() {
                V::A(wa) => (wa.sh.len(), wa.clone()),
                // TODO `wa` doesn't actually need to be a ref counted array
                V::Scalar(ws) => (
                    0,
                    Cc::new(A::new(vec![V::Scalar(*ws)], vec![1], Some(new_scalar(0)))),
                ),
                _ => return Err(Ve::S("dyadic scan w is invalid type")),
            };
            match x.0.unwrap() {
                V::A(xa) => {
                    let s = &xa.sh;
                    if s.is_empty() {
                        return Err(Ve::S("scan dyadic array rank not at least 1"));
                    };
                    if 1 + wr != s.len() {
                        return Err(Ve::S("scan dyadic array rank don't match"));
                    }
                    let l = xa.r.len();
                    let mut r = vec![V::Nothing; l];
                    if l > 0 {
                        let mut c = 1;
                        let mut i = 1;
                        while i < s.len() {
                            c *= s[i];
                            i += 1;
                        }
                        i = 0;
                        while i < c {
                            r[i] = match call(
                                stack,
                                2,
                                Vn(f.0),
                                Vn(Some(&xa.r[i])),
                                Vn(Some(&wa.r[i])),
                            ) {
                                Ok(v) => v.as_v().unwrap().clone(),
                                Err(_e) => panic!("dyadic scan call failed"),
                            };
                            i += 1;
                        }
                        while i < l {
                            r[i] = match call(
                                stack,
                                2,
                                Vn(f.0),
                                Vn(Some(&xa.r[i])),
                                Vn(Some(&r[i - c])),
                            ) {
                                Ok(v) => v.as_v().unwrap().clone(),
                                Err(_e) => panic!("dyadic scan call failed"),
                            };
                            i += 1;
                        }
                    };
                    Ok(Vs::V(V::A(Cc::new(A::new(r, s.to_vec(), xa.fill.clone())))))
                }
                _ => Err(Ve::S("dyadic scan x or w is not an array")),
            }
        }
        _ => panic!("illegal scan arity"),
    }
}
// _fillBy_
fn fill_by(stack: &mut Stack, arity: usize, f: Vn, _g: Vn, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("fill_by");
    call(stack, arity, f, x, w)
}
// ⊘
fn cases(stack: &mut Stack, arity: usize, f: Vn, g: Vn, x: Vn, w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("cases");
    match arity {
        1 => call(stack, arity, f, x, Vn(None)),
        2 => call(stack, arity, g, x, w),
        _ => panic!("illegal cases arity"),
    }
}
// ⎊
fn catches(_stack: &mut Stack, _arity: usize, _f: Vn, _g: Vn, _x: Vn, _w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("catches");
    panic!("catches not implemented");
}

pub fn decompose(arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("decompose");
    #[cfg(feature = "debug-fns")]
    dbg_args("decompose", arity, &x, &Vn(None));
    let r = match arity {
        1 => {
            if
            // atoms
            matches!(
                x.0.as_ref().unwrap(),
                V::Scalar(_) | V::Char(_) | V::Nothing | V::A(_)
            ) {
                Ok(Vs::V(V::A(Cc::new(A::new(
                    vec![V::Scalar(-1.0), x.0.unwrap().clone()],
                    vec![2],
                    None,
                )))))
            } else if
            // primitives
            matches!(
                x.0.as_ref().unwrap(),
                V::BlockInst(_, Some(_))
                    | V::UserMd1(.., Some(_))
                    | V::UserMd2(.., Some(_))
                    | V::Fn(_, Some(_))
                    | V::R1(_, Some(_))
                    | V::R2(_, Some(_))
                    | V::D1(_, Some(_))
                    | V::D2(_, Some(_))
                    | V::Tr2(_, Some(_))
                    | V::Tr3(_, Some(_))
            ) {
                Ok(Vs::V(V::A(Cc::new(A::new(
                    vec![V::Scalar(0.0), x.0.unwrap().clone()],
                    vec![2],
                    None,
                )))))
            } else if
            // repr
            matches!(
                x.0.as_ref().unwrap(),
                V::UserMd1(.., None) | V::UserMd2(.., None)
            ) {
                match x.0.as_ref().unwrap() {
                    V::UserMd1(b, a, None) => {
                        let t = 3 + b.def.typ;
                        match t {
                            4 => {
                                let D1(f, g) = a.deref();
                                Ok(Vs::V(V::A(Cc::new(A::new(
                                    vec![V::Scalar(4.0), g.clone(), f.clone()],
                                    vec![3],
                                    None,
                                )))))
                            }
                            _ => return Err(Ve::S("UserMd1 illegal decompose")),
                        }
                    }
                    V::UserMd2(b, a, None) => {
                        let t = 3 + b.def.typ;
                        match t {
                            5 => {
                                let D2(f, g, h) = a.deref();
                                Ok(Vs::V(V::A(Cc::new(A::new(
                                    vec![V::Scalar(5.0), g.clone(), f.clone(), h.clone()],
                                    vec![4],
                                    None,
                                )))))
                            }
                            _ => return Err(Ve::S("UserMd2 illegal decompose")),
                        }
                    }
                    _ => panic!("decompose other"),
                }
            } else {
                // everything else
                match x.0.as_ref().unwrap() {
                    V::D1(d1, None) => {
                        let D1(m, f) = (*d1).deref();
                        Ok(Vs::V(V::A(Cc::new(A::new(
                            vec![V::Scalar(4.0), f.clone(), m.clone()],
                            vec![3],
                            None,
                        )))))
                    }
                    V::D2(d2, None) => {
                        let D2(m, f, g) = (*d2).deref();
                        Ok(Vs::V(V::A(Cc::new(A::new(
                            vec![V::Scalar(5.0), f.clone(), m.clone(), g.clone()],
                            vec![4],
                            None,
                        )))))
                    }
                    V::Tr2(tr2, None) => {
                        let Tr2(g, h) = (*tr2).deref();
                        Ok(Vs::V(V::A(Cc::new(A::new(
                            vec![V::Scalar(2.0), g.clone(), h.clone()],
                            vec![3],
                            None,
                        )))))
                    }
                    V::Tr3(tr3, None) => {
                        let Tr3(f, g, h) = (*tr3).deref();
                        Ok(Vs::V(V::A(Cc::new(A::new(
                            vec![V::Scalar(3.0), f.clone(), g.clone(), h.clone()],
                            vec![4],
                            None,
                        )))))
                    }
                    _ => Ok(Vs::V(V::A(Cc::new(A::new(
                        vec![V::Scalar(1.0), x.0.unwrap().clone()],
                        vec![2],
                        None,
                    ))))),
                }
            }
        }
        _ => panic!("illegal decompose arity"),
    };
    #[cfg(feature = "debug-fns")]
    dbg_rtn("decompose", arity, &r);
    r
}

pub fn prim_ind(arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    #[cfg(feature = "coz-fns")]
    coz::scope!("prim_ind");
    match arity {
        1 => match x.0.unwrap() {
            V::BlockInst(_b, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::UserMd1(_b, _a, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::UserMd2(_b, _a, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::Fn(_a, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::R1(_f, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::R2(_f, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::D1(_d1, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::D2(_d2, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::Tr2(_tr2, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            V::Tr3(_tr3, Some(prim)) => Ok(Vs::V(V::Scalar(*prim as f64))),
            _ => Ok(Vs::V(V::Scalar(64_f64))),
        },
        _ => panic!("illegal plus arity"),
    }
}

pub fn glyph(_arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    let glyphs = "+-×÷⋆√⌊⌈|¬∧∨<>≠=≤≥≡≢⊣⊢⥊∾≍⋈↑↓↕«»⌽⍉/⍋⍒⊏⊑⊐⊒∊⍷⊔!˙˜˘¨⌜⁼´˝`∘○⊸⟜⌾⊘◶⎉⚇⍟⎊";
    let fmt = match x.0.unwrap() {
        V::BlockInst(_b, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::UserMd1(_b, _a, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::UserMd2(_b, _a, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::Fn(_a, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::R1(_f, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::R2(_f, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::D1(_d1, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::D2(_d2, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::Tr2(_tr2, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        V::Tr3(_tr3, Some(prim)) => new_char(glyphs.chars().nth(*prim).unwrap()),
        xv => new_string(&format!("{}", xv)),
    };
    Ok(Vs::V(fmt))
}

pub fn fmtnum(arity: usize, x: Vn, _w: Vn) -> Result<Vs, Ve> {
    match arity {
        1 => match x.0.unwrap() {
            V::Scalar(n) => Ok(Vs::V(new_string(&format!("{}", *n)))),
            _ => Err(Ve::S("no matching value for fmtnum")),
        },
        _ => panic!("typ arity not implemented"),
    }
}

pub fn provide() -> A {
    let fns = vec![
        V::Fn(Fn(typ), None),
        V::Fn(Fn(fill), None),
        V::Fn(Fn(log), None),
        V::Fn(Fn(group_len), None),
        V::Fn(Fn(group_ord), None),
        V::Fn(Fn(assert_fn), None),
        V::Fn(Fn(plus), None),
        V::Fn(Fn(minus), None),
        V::Fn(Fn(times), None),
        V::Fn(Fn(divide), None),
        V::Fn(Fn(power), None),
        V::Fn(Fn(floor), None),
        V::Fn(Fn(equals), None),
        V::Fn(Fn(lesseq), None),
        V::Fn(Fn(shape), None),
        V::Fn(Fn(reshape), None),
        V::Fn(Fn(pick), None),
        V::Fn(Fn(windows), None),
        V::R1(R1(table), None),
        V::R1(R1(scan), None),
        V::R2(R2(fill_by), None),
        V::R2(R2(cases), None),
        V::R2(R2(catches), None),
    ];
    A::new(fns, vec![23], None)
}
