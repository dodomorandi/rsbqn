use crate::schema::{Block, BlockInst, Code, EnvUnboxed, Tr2, Tr3, A, D1, D2, V};
use bacon_rajan_cc::{Trace, Tracer /*collect_cycles*/};

impl Trace for V {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing V");
    }
}

impl Trace for Code {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing Code");
    }
}

impl Trace for Block {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing Code");
    }
}

impl Trace for EnvUnboxed {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing env");
    }
}

impl Trace for Tr2 {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing tr2");
    }
}

impl Trace for Tr3 {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing tr3");
    }
}

impl Trace for D1 {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing d1");
    }
}

impl Trace for D2 {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing d2");
    }
}

impl Trace for BlockInst {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing blockinst");
    }
}

impl Trace for A {
    fn trace(&self, _tracer: &mut Tracer) {
        panic!("clearing array");
    }
}
