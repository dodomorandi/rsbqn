-module(gen_test).
-mode(compile).
-export([main/1]).

% usage:
% escript gen_code.erl path/to/bqn

% https://stackoverflow.com/a/2249387
cmd(Cmd, Args) ->
    Tag = make_ref(),
    {Pid, Ref} = erlang:spawn_monitor(fun() ->
            Rv = cmd_sync(Cmd, Args),
            exit({Tag, Rv})
        end),
    receive
        {'DOWN', Ref, process, Pid, {Tag, Data}} -> Data;
        {'DOWN', Ref, process, Pid, Reason} -> exit(Reason)
    end.

cmd_sync(Cmd, Args) ->
    P = open_port({spawn_executable, os:find_executable(Cmd)}, [
            binary, use_stdio, stream, eof, {args, Args}]),
    cmd_receive(P, []).

cmd_receive(Port, Acc) ->
    receive
        {Port, {data, Data}} -> cmd_receive(Port, [Data|Acc]);
        {Port, eof}          -> {ok, lists:reverse(Acc)}
    end.

r0(Repo) ->
    {ok,Cwd} = file:get_cwd(),
    {Cmd,Args} = { erlang:binary_to_list(filename:join([Cwd, <<"crs.bqn">>])),[Repo,<<"r0">>] },
    cmd(Cmd,Args).
    
main([Repo]) ->
    {ok,R0} = r0(Repo),
    io:format("~ts~n",[erlang:iolist_to_binary([
        <<"use log::{debug};\n">>,
        <<"use core::f64::{INFINITY,NEG_INFINITY};\n">>,
        <<"use crate::ebqn::{run,assert_panic};\n">>,
        <<"use crate::schema::{Code,new_scalar,new_char,new_string,Body,A,Decoder,V};\n">>,
        <<"pub fn r0(provide: A) -> V {\nrun(Code::new(">>,R0,<<"))}\n\n">>
    ])]);
main(_Args) ->
    io:format("bad arguments~n"),
    halt(1).