#![forbid(unsafe_code)]

mod late_init;
pub mod provide;
pub mod schema;
mod trace;
pub mod vm;
pub mod gen {
    pub mod code;
}
pub mod fmt;

// https://docs.rs/env_logger/0.7.1/env_logger/#capturing-logs-in-tests
pub fn init_log() {
    let _ = env_logger::builder().is_test(true).try_init();
}

macro_rules! coz_scope {
    ($name:literal, $block:block) => {
        #[cfg(feature = "coz-ops")]
        {
            ::coz::begin!($name);
            let out = $block;
            ::coz::end!($name);
            out
        }

        #[cfg(not(feature = "coz-ops"))]
        $block
    };
}
pub(crate) use coz_scope;
