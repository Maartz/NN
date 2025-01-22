-module(helpers).
-compile(export_all).

generate_ids(0, Acc) ->
    Acc;
generate_ids(Index, Acc) ->
    Id = generate_id(),
    generate_ids(Index - 1, [Id | Acc]).

generate_id() ->
    UniqueInt = erlang:unique_integer([positive, monotonic]),
    1.0 / (UniqueInt + 1). 