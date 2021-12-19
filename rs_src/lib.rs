mod schema;
mod late_init;
mod trace;
mod ebqn;
mod prim;
mod test;
mod code;
mod fmt;
use rustler::{Env,Term};
//use syslog::Facility;
extern crate log_panics;

// https://docs.rs/env_logger/0.7.1/env_logger/#capturing-logs-in-tests
pub fn init_log() {
    let _ = env_logger::builder().is_test(true).try_init();
}

pub fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(schema::Env, env);
    //let _r = syslog::init(Facility::LOG_USER,
    //             log::LevelFilter::Info,
    //             Some("ebqn"));
    log_panics::init();
    true
}
rustler::init!("ebqn", [ebqn::init_st],load=load);
