#!/usr/bin/env escript
main(Args) ->
    Root = filename:dirname(filename:dirname(filename:absname(escript:script_name()))),
    true = code:add_patha(filename:join([Root, "_build", "default", "lib", "nn", "ebin"])),
    Path = case Args of
        [] -> filename:join([Root, "_build", "life", "index.html"]);
        [Output] -> Output;
        _ -> io:format(standard_error, "Usage: ./scripts/life.sh [output.html]~n", []), halt(2)
    end,
    try life_demo:export(Path)
    catch Class:Reason:Stack ->
        io:format(standard_error, "Life simulation failed: ~p:~p~n~p~n", [Class, Reason, Stack]),
        halt(1)
    end.
