%% Export real Erlang episodes to a self-contained, offline browser replay.
-module(life_demo).
-export([export/1, export/2]).

export(Path) -> export(Path, #{}).
export(Path, Options) ->
    Experiment = life_evolution:run(Options),
    ok = filelib:ensure_dir(Path),
    Dir = filename:dirname(Path),
    ExoFile = filename:join(Dir, "exoself.ets"),
    Exo = life_exoself:compare(Experiment, ExoFile),
    io:format("ExoSelf: score ~.2f | ~w evaluations incl. ~w initial | ~w cycles of tuning~n",
              [maps:get(score,Exo),maps:get(total_evaluations,Exo),
               maps:get(initial_evaluations,Exo),maps:get(cycles,Exo)]),
    ValidationSeeds = [101,202,303],
    Worlds = [#{seed => Seed,
                scripted => life_world:episode(scripted, Seed),
                initial => life_world:episode(maps:get(initial, Experiment), Seed),
                evolved => life_world:episode(maps:get(champion, Experiment), Seed),
                exoself => life_exoself:episode(ExoFile, Seed)}
              || Seed <- ValidationSeeds],
    Data = Experiment#{worlds => Worlds, validation_seeds => ValidationSeeds, exoself => Exo},
    {ok, Template} = file:read_file(filename:join([code:priv_dir(nn), "life", "index.html"])),
    Encoded = iolist_to_binary(json(Data)),
    Html = binary:replace(Template, <<"__LIFE_DATA__">>, Encoded),
    ok = filelib:ensure_dir(Path),
    ok = file:write_file(Path, Html),
    ok = file:write_file(filename:join(Dir, "experiment.json"), Encoded),
    ok = file:write_file(filename:join(Dir, "champion.term"),
                         io_lib:format("~p.~n", [maps:with([champion, seed, training_seeds], Experiment)])),
    [io:format("Validation ~p: programmed=~w, initial=~w, population=~w, exoself=~w foods~n",
               [maps:get(seed, W), maps:get(eaten, maps:get(scripted, W)),
                maps:get(eaten, maps:get(initial, W)), maps:get(eaten, maps:get(evolved, W)),
                maps:get(eaten, maps:get(exoself, W))])
     || W <- Worlds],
    io:format("Replay: ~ts~n", [filename:absname(Path)]),
    ok.

%% Data contains only fixed atom keys/labels, numbers, arrays and maps.
%% Binaries use escaping so this also remains valid when labels are extended.
json(Map) when is_map(Map) ->
    ["{", lists:join(",", [[json(K), ":", json(V)] || {K,V} <- maps:to_list(Map)]), "}"];
json(List) when is_list(List) -> ["[", lists:join(",", [json(V) || V <- List]), "]"];
json(Tuple) when is_tuple(Tuple) -> json(tuple_to_list(Tuple));
json(Atom) when is_atom(Atom) -> json(atom_to_binary(Atom, utf8));
json(Binary) when is_binary(Binary) ->
    ["\"", [escape(C) || C <- binary_to_list(Binary)], "\""];
json(Integer) when is_integer(Integer) -> integer_to_binary(Integer);
json(Float) when is_float(Float) -> float_to_binary(Float, [short]).

escape($") -> "\\\"";
escape($\\) -> "\\\\";
escape(C) when C < 32 -> io_lib:format("\\u~4.16.0B", [C]);
escape($<) -> "\\u003c";
escape(C) -> C.
