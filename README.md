rsbqn
=====

RSBQN is a Rust implementation of a BQN virtual machine.

Build
------

    cargo build --features repl

Status
------

|TEST|STATUS|DETAIL
|---|---|---|
|bytecode|ok|37 passed; 0 failed; 0 ignored;|
|identity|ok|14 passed; 0 failed; 0 ignored;|
|prim|ok|556 passed; 0 failed; 0 ignored;|
|simple|ok|20 passed; 0 failed; 0 ignored;|
|under|FAILED|40 passed; 1 failed; 0 ignored;|
|undo|ok|68 passed; 0 failed; 0 ignored;|
|fill|N/A|N/A|
|header|FAILED|146 passed; 4 failed; 0 ignored;|
|literal|ok|52 passed; 0 failed; 0 ignored;|
|namespace|N/A|N/A|
|syntax|ok|135 passed; 0 failed; 0 ignored;|
|token|ok|27 passed; 0 failed; 0 ignored;|
|unhead|N/A|N/A|

Test
-----

    escript gen_test.erl /path/to/mlochbaum/bqn
    cargo gen-runtime /path/to/mlochbaum/bqn

    cargo test

Causal Profiling
-----

    cargo build --profile bench --features coz-loop,coz-ops,coz-fns
    coz run --- ./target/release/rsbqn

Heap Analysis
-----

    cargo build --profile bench --features dhat
    ./target/release/rsbqn

Perf
-----

    cargo build --profile bench
    perf stat -e instructions ./target/release/rsbqn

Profile Guided Optimiziation
-----

    RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release --target=x86_64-unknown-linux-gnu
    for i in {1..10}; do ./target/x86_64-unknown-linux-gnu/release/rsbqn ; done
    ${HOME}/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/
    RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release --target=x86_64-unknown-linux-gnu
