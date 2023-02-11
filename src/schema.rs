use crate::late_init::LateInit;
use crate::vm::vm;
use bacon_rajan_cc::Cc;
use enum_as_inner::EnumAsInner;

use std::cell::RefCell;
use std::fmt;
use std::ops::Deref;

// Traits
pub trait Calleable {
    fn call(&self, stack: &mut Stack, arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve>;
}
pub trait Decoder {
    fn to_f64(&self) -> f64;
}
pub trait Stacker {
    fn pop_list(&mut self, n: usize) -> Vec<V>;
    fn pop_ref_list(&mut self, n: usize) -> Vec<Vs>;
}

#[derive(Clone)]
pub struct Fn(pub fn(usize, Vn, Vn) -> Result<Vs, Ve>);

impl fmt::Debug for Fn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Fn").field(&"fn").finish()
    }
}

impl PartialEq for Fn {
    fn eq(&self, other: &Self) -> bool {
        self.0 as usize == other.0 as usize
    }
}

#[derive(Clone)]
pub struct R1(pub fn(&mut Stack, usize, Vn, Vn, Vn) -> Result<Vs, Ve>);

impl fmt::Debug for R1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("R1").field(&"fn").finish()
    }
}

impl PartialEq for R1 {
    fn eq(&self, other: &Self) -> bool {
        self.0 as usize == other.0 as usize
    }
}

type R2Fn = fn(&mut Stack, usize, Vn, Vn, Vn, Vn) -> Result<Vs, Ve>;

#[derive(Clone)]
pub struct R2(pub R2Fn);

impl fmt::Debug for R2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("R2").field(&"fn").finish()
    }
}

impl PartialEq for R2 {
    fn eq(&self, other: &Self) -> bool {
        self.0 as usize == other.0 as usize
    }
}

// Value (unboxed)
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum V {
    Scalar(f64),
    Char(char),
    BlockInst(Cc<BlockInst>, Option<usize>),
    UserMd1(Cc<BlockInst>, Cc<D1>, Option<usize>),
    UserMd2(Cc<BlockInst>, Cc<D2>, Option<usize>),
    Nothing,
    A(Cc<A>),
    Fn(Fn, Option<usize>),     // X, W
    R1(R1, Option<usize>),     // F, X, W
    R2(R2, Option<usize>),     // F, G, X, W
    D1(Cc<D1>, Option<usize>), // M, F
    D2(Cc<D2>, Option<usize>), // M, F, G
    Tr2(Cc<Tr2>, Option<usize>),
    Tr3(Cc<Tr3>, Option<usize>),
}
impl V {
    pub fn is_fn(&self) -> bool {
        matches!(
            self,
            V::BlockInst(..) | V::UserMd1(..) | V::UserMd2(..) | V::Tr2(..) | V::Tr3(..)
        )
    }
}
impl Decoder for i64 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}
impl Decoder for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }
}
impl Decoder for V {
    fn to_f64(&self) -> f64 {
        match self {
            V::Scalar(n) => *n,
            V::Char(c) => f64::from(u32::from(*c)),
            V::BlockInst(_b, _prim) => panic!("can't decode blockinst to RUST"),
            V::UserMd1(_b, _a, _prim) => panic!("can't encode UserMd1 to RUST"),
            V::UserMd2(_b, _a, _prim) => panic!("can't encode UserMd2 to RUST"),
            V::Nothing => panic!("can't decode nothing to RUST"),
            V::A(_a) => panic!("can't decode array to RUST"),
            V::Fn(_a, _prim) => panic!("can't decode fn to RUST"),
            V::R1(_f, _prim) => panic!("can't decode r1 to RUST"),
            V::R2(_f, _prim) => panic!("can't decode r2 to RUST"),
            V::D1(_d1, _prim) => panic!("can't decode d1 to RUST"),
            V::D2(_d2, _prim) => panic!("can't decode d2 to RUST"),
            V::Tr2(_tr2, _prim) => panic!("can't decode train2 to RUST"),
            V::Tr3(_tr3, _prim) => panic!("can't decode train3 to RUST"),
        }
    }
}
impl Calleable for V {
    fn call(&self, stack: &mut Stack, arity: usize, x: Vn, w: Vn) -> Result<Vs, Ve> {
        match self {
            V::UserMd1(b, mods, _prim) => {
                let D1(m, f) = mods.deref();
                let args = vec![
                    Some(self.clone()),
                    x.none_or_clone(),
                    w.none_or_clone(),
                    Some(m.clone()),
                    Some(f.clone()),
                ];
                let env = Env::new(Some(&b.parent), &b.def, arity, Some(args));
                let pos = body_pos(b, arity);
                let (bodies, body_id) = bodies(b, arity);
                vm(&env, &b.def.code, bodies, body_id, pos, stack)
            }
            V::UserMd2(b, mods, _prim) => {
                let D2(m, f, g) = mods.deref();
                let args = vec![
                    Some(self.clone()),
                    x.none_or_clone(),
                    w.none_or_clone(),
                    Some(m.clone()),
                    Some(f.clone()),
                    Some(g.clone()),
                ];
                let env = Env::new(Some(&b.parent), &b.def, arity, Some(args));
                let pos = body_pos(b, arity);
                let (bodies, body_id) = bodies(b, arity);
                vm(&env, &b.def.code, bodies, body_id, pos, stack)
            }
            V::BlockInst(b, _prim) => {
                let args = vec![Some(self.clone()), x.none_or_clone(), w.none_or_clone()];
                let env = Env::new(Some(&b.parent), &b.def, arity, Some(args));
                let pos = body_pos(b, arity);
                let (bodies, body_id) = bodies(b, arity);
                vm(&env, &b.def.code, bodies, body_id, pos, stack)
            }
            V::Scalar(n) => Ok(Vs::V(V::Scalar(*n))),
            V::Char(c) => Ok(Vs::V(V::Char(*c))),
            V::Fn(f, _prim) => f.0(arity, x, w),
            V::R1(_f, _prim) => panic!("can't call r1"),
            V::R2(_f, _prim) => panic!("can't call r2"),
            V::D1(d1, _prim) => {
                let D1(m, f) = d1.deref();
                let r = match m {
                    V::R1(r1, _prim) => r1.0(stack, arity, Vn(Some(f)), x, w),
                    _ => panic!("can only call raw1 mods in derv1"),
                };
                r
            }
            V::D2(d2, _prim) => {
                let D2(m, f, g) = d2.deref();
                match m {
                    V::R2(r2, _prim) => r2.0(stack, arity, Vn(Some(f)), Vn(Some(g)), x, w),
                    _ => panic!("can only call raw2 mods in derv2"),
                }
            }
            V::Tr2(tr, _prim) => {
                let Tr2(g, h) = tr.deref();
                match h.call(stack, arity, x, w) {
                    Ok(r) => g.call(stack, 1, Vn(*Some(&r.as_v()).unwrap()), Vn(None)),
                    Err(e) => Err(e),
                }
            }
            V::Tr3(tr, _prim) => {
                let Tr3(f, g, h) = tr.deref();
                match h.call(stack, arity, Vn(x.0), Vn(w.0)) {
                    Ok(r) => match f.call(stack, arity, Vn(x.0), Vn(w.0)) {
                        Ok(l) => g.call(
                            stack,
                            2,
                            Vn(Some(r.as_v().unwrap())),
                            Vn(Some(l.as_v().unwrap())),
                        ),
                        Err(e) => Err(e),
                    },
                    Err(e) => Err(e),
                }
            }
            V::A(_) => Ok(Vs::V(self.clone())),
            V::Nothing => Ok(Vs::V(V::Nothing)),
        }
    }
}

// Value (Optional)
pub struct Vn<'a>(pub Option<&'a V>);
impl<'a> Vn<'a> {
    fn none_or_clone(&self) -> Vh {
        match self.deref().0 {
            None => Some(V::Nothing),
            Some(v) => Some(v.clone()),
        }
    }
}

// Value (boxed on the stack)
#[derive(Debug, Clone, EnumAsInner)]
pub enum Vs {
    V(V),
    Slot(Env, usize),
    Ar(Ar),
    Match(Option<V>),
    Nothing,
}
impl Vs {
    pub fn get(&self) -> V {
        match self {
            Vs::Slot(env, id) => env.get(*id),
            Vs::Ar(a) => {
                let shape = vec![a.r.len()];
                let ravel =
                    a.r.iter()
                        .map(|e| match e {
                            Vs::Slot(env, id) => env.get(*id),
                            _ => panic!("ref array contains a non-slot"),
                        })
                        .collect::<Vec<V>>();
                V::A(Cc::new(A::new(ravel, shape, None)))
            }
            _ => panic!("can only resolve slots or ref arrays"),
        }
    }
    pub fn set(&self, d: bool, v: &V) -> Result<V, Ve> {
        match self {
            Vs::Slot(env, id) => {
                env.set(d, *id, v)?;
                Ok(v.clone())
            }
            Vs::Ar(a) => match v {
                V::A(va) => {
                    if va.sh != a.sh {
                        Err(Ve::S("target and value shapes don't match"))
                    } else {
                        for i in 0..a.r.len() {
                            let _ = &a.r[i].set(d, &va.r[i])?;
                        }
                        Ok(v.clone())
                    }
                }
                _ => Err(Ve::S("")),
            },
            Vs::Match(mb) => match mb {
                Some(m) => {
                    if m != v {
                        Err(Ve::S(""))
                    } else {
                        Ok(v.clone())
                    }
                }
                None => Ok(v.clone()),
            },
            _ => panic!("can only set slots"),
        }
    }
}
impl Default for Vs {
    fn default() -> Self {
        Vs::Nothing
    }
}

// Value (boxed on the heap)
pub type Vh = Option<V>;

// Value error
#[derive(Debug)]
pub enum Ve {
    S(&'static str),
    V(V),
}

// Stack
#[derive(Debug)]
pub struct Stack {
    pub s: Vec<Vs>,
    pub fp: usize,
}
impl Stack {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self {
            s: Vec::with_capacity(128),
            fp: 0,
        }
    }
}
impl Stacker for Vec<Vs> {
    fn pop_list(&mut self, x: usize) -> Vec<V> {
        let len = self.len();
        self.drain((len - x)..)
            .map(|vs| vs.into_v().unwrap())
            .collect()
    }
    fn pop_ref_list(&mut self, x: usize) -> Vec<Vs> {
        let len = self.len();
        self.drain((len - x)..).collect()
    }
}

// Bodies
#[derive(Debug, Clone, PartialEq)]
pub enum Bodies {
    Comp(usize),                 // compressed
    Head(Vec<usize>),            // compressed with header
    Exp(Vec<usize>, Vec<usize>), // expanded
}

//#[derive(Debug,Clone,PartialEq)]
//pub struct Exp(pub Vec<usize>,pub Vec<usize>); // Monadic, Dyadic

// Code
#[derive(Default, Debug, PartialEq)]
pub struct Code {
    pub bc: Vec<usize>,
    pub objs: Vec<V>,
    pub body_ids: Vec<(usize, usize)>,
    pub blocks: LateInit<Vec<Cc<Block>>>,
}
impl Code {
    pub fn new(
        bc: Vec<usize>,
        objs: Vec<V>,
        blocks_raw: Vec<(u8, bool, Bodies)>,
        body_ids: Vec<(usize, usize)>,
    ) -> Cc<Self> {
        let code = Cc::new(Self {
            bc,
            objs,
            body_ids,
            blocks: LateInit::default(),
        });
        let blocks_derv = blocks_raw
            .into_iter()
            .map(|block| {
                let (typ, imm, bodies) = block;
                let b = Block {
                    typ,
                    imm,
                    bodies,
                    code: LateInit::default(),
                };
                b.code.init(code.clone());
                Cc::new(b)
            })
            .collect::<Vec<Cc<Block>>>();
        code.blocks.init(blocks_derv);
        code
    }
}

// Block
#[derive(Debug, PartialEq)]
pub struct Block {
    pub typ: u8,
    pub imm: bool,
    pub bodies: Bodies,
    pub code: LateInit<Cc<Code>>,
}

// Env (Unboxed)
#[derive(Default, Debug)]
pub struct EnvUnboxed {
    pub parent: Option<Env>,
    pub vars: RefCell<Vec<Vh>>,
    pub num_args: usize,
    pub init_args: Option<Vec<Vh>>,
}

#[derive(Clone, Debug)]
pub struct Env(Cc<EnvUnboxed>);
impl Env {
    pub fn new(
        parent: Option<&Env>,
        block: &Cc<Block>,
        arity: usize,
        args: Option<Vec<Vh>>,
    ) -> Self {
        let (_pos, locals) = match &block.bodies {
            Bodies::Comp(b) => *block.code.body_ids.get(*b).unwrap(),
            Bodies::Head(amb) => *block.code.body_ids.get(amb[0]).unwrap(),
            Bodies::Exp(mon, dya) => match arity {
                1 => *block.code.body_ids.get(mon[0]).unwrap(),
                2 => *block.code.body_ids.get(dya[0]).unwrap(),
                n => panic!("invalid arity for deferred block {}", n),
            },
        };
        let (vars, num_args) = match &args {
            None => {
                let mut v: Vec<Vh> = Vec::with_capacity(locals);
                v.resize_with(locals, || None);
                (v, 0)
            }
            Some(v) => {
                let n = v.len();
                let mut s = v.clone();
                s.resize_with(locals, || None);
                (s, n)
            }
        };
        let env = EnvUnboxed {
            parent: parent.cloned(),
            vars: RefCell::new(vars),
            num_args,
            init_args: args,
        };
        Self(Cc::new(env))
    }
    pub fn reinit(&self, locals: usize) -> Self {
        match self {
            Env(env) => {
                let vars = match &env.init_args {
                    None => {
                        let mut v: Vec<Vh> = Vec::with_capacity(locals);
                        v.resize_with(locals, || None);
                        v
                    }
                    Some(v) => {
                        let mut s = v.clone();
                        s.resize_with(locals, || None);
                        s
                    }
                };
                Self(Cc::new(EnvUnboxed {
                    parent: env.parent.clone(),
                    vars: RefCell::new(vars),
                    num_args: env.num_args,
                    init_args: env.init_args.clone(),
                }))
            }
        }
    }
    pub fn new_root() -> Self {
        let env = EnvUnboxed {
            parent: None,
            vars: RefCell::new(vec![]),
            num_args: 0,
            init_args: None,
        };
        Self(Cc::new(env))
    }

    pub fn get(&self, id: usize) -> V {
        match self {
            Env(e) => match e.vars.borrow().get(id).unwrap() {
                Some(v) => v.clone(),
                None => panic!("heap slot is undefined"),
            },
        }
    }
    pub fn set(&self, d: bool, id: usize, v: &V) -> Result<(), Ve> {
        match self {
            Env(e) => {
                let slot_is_none = e.vars.borrow().get(id).unwrap().is_none();
                match d == slot_is_none {
                    false => Err(Ve::S("unexpected slot value during assignment")),
                    true => {
                        let v = v.clone();
                        *e.vars.borrow_mut().get_mut(id).unwrap() = Some(v);
                        Ok(())
                    }
                }
            }
        }
    }
    pub fn get_drop(&self, id: usize) -> V {
        match self {
            Env(e) => {
                let r = match e.vars.borrow().get(id).unwrap() {
                    Some(v) => v.clone(),
                    None => panic!("heap slot is undefined"),
                };
                *e.vars.borrow_mut().get_mut(id).unwrap() = None;
                r
            }
        }
    }
    pub fn ge(&self, mut i: usize) -> &Env {
        let mut cur = self;
        loop {
            match i {
                0 => break cur,
                _ => {
                    i -= 1;
                    cur = cur.0.parent.as_ref().unwrap();
                }
            }
        }
    }
    pub fn extend(&self, vars: usize) {
        match self {
            Env(e) => {
                e.vars
                    .borrow_mut()
                    .extend(std::iter::repeat(None).take(vars));
            }
        }
    }
    pub fn to_vars(&self) -> V {
        match self {
            Env(e) => {
                let ravel = e
                    .vars
                    .borrow()
                    .iter()
                    .map(|e| match e {
                        Some(s) => s.clone(),
                        None => V::Nothing,
                    })
                    .collect::<Vec<V>>();
                let shape = vec![ravel.len()];
                V::A(Cc::new(A::new(ravel, shape, None)))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BlockInst {
    pub def: Cc<Block>,
    parent: Env,
}
impl BlockInst {
    pub fn new(env: Env, block: Cc<Block>) -> Self {
        Self {
            def: block,
            parent: env,
        }
    }
    pub fn call_md1(&self, stack: &mut Stack, arity: usize, args: D1) -> Result<Vs, Ve> {
        match self.def.imm {
            false => {
                let r = Vs::V(V::UserMd1(
                    Cc::new(BlockInst::new(self.parent.clone(), self.def.clone())),
                    Cc::new(args),
                    None,
                ));
                Ok(r)
            }
            true => {
                let (pos, bodies, body_id) = match &self.def.bodies {
                    Bodies::Comp(b) => {
                        let (p, _l) = self.def.code.body_ids[*b];
                        (p, None, None)
                    }
                    Bodies::Head(amb) => {
                        let (p, _l) = self.def.code.body_ids[amb[0]];
                        (p, Some(amb), Some(0))
                    }
                    _ => panic!("body immediacy doesnt match block definition"),
                };
                let D1(m, f) = args;
                let env = Env::new(
                    Some(&self.parent),
                    &self.def,
                    arity,
                    Some(vec![Some(m), Some(f)]),
                );
                vm(&env, &self.def.code, bodies, body_id, pos, stack)
            }
        }
    }
    pub fn call_md2(&self, stack: &mut Stack, arity: usize, args: D2) -> Result<Vs, Ve> {
        match self.def.imm {
            false => {
                let r = Vs::V(V::UserMd2(
                    Cc::new(BlockInst::new(self.parent.clone(), self.def.clone())),
                    Cc::new(args),
                    None,
                ));
                Ok(r)
            }
            true => {
                let pos = match self.def.bodies {
                    Bodies::Comp(b) => {
                        let (p, _l) = self.def.code.body_ids[b];
                        p
                    }
                    _ => panic!("body immediacy doesnt match block definition"),
                };
                let D2(m, f, g) = args;
                let env = Env::new(
                    Some(&self.parent),
                    &self.def,
                    arity,
                    Some(vec![Some(m), Some(f), Some(g)]),
                );
                vm(&env, &self.def.code, None, None, pos, stack)
            }
        }
    }
}
impl PartialEq for BlockInst {
    fn eq(&self, other: &Self) -> bool {
        Cc::ptr_eq(&self.def, &other.def) && Cc::ptr_eq(&self.parent.0, &other.parent.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct A {
    pub r: Vec<V>,
    pub sh: Vec<usize>,
    pub fill: Option<V>,
}
impl A {
    pub fn new(r: Vec<V>, sh: Vec<usize>, fill: Option<V>) -> Self {
        Self { r, sh, fill }
    }
}

// array of references. rank-1 for now.
#[derive(Debug, Clone)]
pub struct Ar {
    r: Vec<Vs>,
    sh: Vec<usize>,
}
impl Ar {
    pub fn new(r: Vec<Vs>) -> Self {
        let sh = vec![r.len()];
        Self { r, sh }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct D1(pub V, pub V);
impl D1 {
    pub fn new(m: V, f: V) -> Self {
        Self(m, f)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct D2(pub V, pub V, pub V);
impl D2 {
    pub fn new(m: V, f: V, g: V) -> Self {
        Self(m, f, g)
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Tr2(pub V, pub V);
impl Tr2 {
    pub fn new(g: Vs, h: Vs) -> Self {
        Self(g.into_v().unwrap(), h.into_v().unwrap())
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Tr3(pub V, pub V, pub V);
impl Tr3 {
    pub fn new(f: Vs, g: Vs, h: Vs) -> Self {
        Self(
            f.into_v().unwrap(),
            g.into_v().unwrap(),
            h.into_v().unwrap(),
        )
    }
}

pub struct Runtime(pub A);
pub struct Compiler(pub Cc<BlockInst>);
pub struct Prog(pub Cc<Code>);

// Utility fns
pub fn new_scalar<T: Decoder>(n: T) -> V {
    V::Scalar(n.to_f64())
}
pub fn body_pos(b: &Cc<BlockInst>, arity: usize) -> usize {
    let (pos, _locals) = match &b.def.bodies {
        Bodies::Comp(body) => b.def.code.body_ids[*body],
        Bodies::Head(amb) => b.def.code.body_ids[amb[0]],
        Bodies::Exp(mon, dya) => match arity {
            1 => b.def.code.body_ids[mon[0]],
            2 => b.def.code.body_ids[dya[0]],
            _ => panic!("bad call arity"),
        },
    };
    pos
}
pub fn bodies(b: &Cc<BlockInst>, arity: usize) -> (Option<&Vec<usize>>, Option<usize>) {
    match &b.def.bodies {
        Bodies::Comp(_body) => (None, None),
        Bodies::Head(amb) => (Some(amb), Some(0)),
        Bodies::Exp(mon, dya) => match arity {
            1 => (Some(mon), Some(0)),
            2 => (Some(dya), Some(0)),
            _ => panic!("bad call arity"),
        },
    }
}
pub fn new_char(n: char) -> V {
    V::Char(n)
}
pub fn new_string(n: &str) -> V {
    let ravel = n.to_string().chars().map(V::Char).collect::<Vec<V>>();
    let shape = vec![ravel.len()];
    V::A(Cc::new(A::new(ravel, shape, Some(new_char(' ')))))
}
