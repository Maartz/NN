-module(life_tests).
-include_lib("eunit/include/eunit.hrl").

world_seed_test() ->
    W = life_world:new(42),
    ?assertEqual(W, life_world:new(42)),
    ?assertNotEqual(W, life_world:new(43)),
    Food = maps:get(food, W),
    ?assertEqual(14, length(lists:usort(Food))),
    ?assertNot(lists:member(maps:get(position, W), Food)),
    ?assertEqual(7, length(life_world:sense(W))).

food_and_wall_test() ->
    W = (life_world:new(42))#{position => {0,0}, food => [{1,0},{2,0}], energy => 55},
    Wall = life_world:step(W, west),
    ?assertEqual({0,0}, maps:get(position, Wall)),
    ?assertEqual(54, maps:get(energy, Wall)),
    ?assertEqual(wall, maps:get(event, Wall)),
    Ate = life_world:step(W, east),
    ?assertEqual(60, maps:get(energy, Ate)),
    ?assertEqual(1, maps:get(eaten, Ate)),
    ?assertEqual([{2,0}], maps:get(food, Ate)),
    ?assertEqual(ate, maps:get(event, Ate)).

termination_test() ->
    W = (life_world:new(42))#{position => {0,0}, food => [{3,3}], energy => 1},
    Dead = life_world:step(W, west),
    ?assertEqual(exhausted, maps:get(status, Dead)),
    ?assertEqual(Dead, life_world:step(Dead, east)),
    Fed = life_world:step(W#{food => [{1,0}]}, east),
    ?assertEqual(cleared, maps:get(status, Fed)),
    ?assertEqual(18, maps:get(energy, Fed)),
    Limited = life_world:step(W#{energy => 30, tick => 159}, east),
    ?assertEqual(limit, maps:get(status, Limited)).

scripted_episode_test() ->
    A = life_world:episode(scripted, 101),
    ?assertEqual(A, life_world:episode(scripted, 101)),
    ?assertEqual(14, maps:get(eaten, A)),
    ?assertEqual(cleared, maps:get(status, A)),
    ?assertEqual(maps:get(steps, A) + 1, length(maps:get(frames, A))).

brain_test() ->
    rand:seed(exsplus, 42),
    B = life_brain:new(),
    Inputs = life_world:sense(life_world:new(42)),
    Outputs = life_brain:forward(B, Inputs),
    ?assertEqual(4, length(Outputs)),
    ?assert(lists:all(fun(X) -> X >= -1 andalso X =< 1 end, Outputs)),
    ?assert(lists:member(life_brain:decide(B, Inputs), [north,east,south,west])),
    ?assertEqual(76, length(life_brain:weights(B))),
    ?assertNotEqual(B, life_brain:mutate(B)),
    ?assertEqual(life_world:episode(B, 101), life_world:episode(B, 101)).


evolution_test_() -> {timeout, 10, fun() ->
    Options = #{seed => 42, population => 8, generations => 3, verbose => false},
    A = life_evolution:run(Options),
    ?assertEqual(A, life_evolution:run(Options)),
    History = maps:get(history, A),
    Scores = [maps:get(score, G) || G <- History],
    ?assertEqual(4, length(History)),
    ?assertEqual(lists:sort(Scores), Scores),
    ?assertEqual([11,22,33], maps:get(training_seeds, A)),
    ?assertEqual(76, length(life_brain:weights(maps:get(champion, A))))
end}.

worker_error_test() ->
    ?assertException(error, {evaluation_failed, _},
                     life_evolution:evaluate_population([#{}], [11,22,33])).

default_learning_test_() -> {timeout, 10, fun() ->
    Run = life_evolution:run(#{verbose => false}),
    Initial = maps:get(initial, Run),
    Champion = maps:get(champion, Run),
    History = maps:get(history, Run),
    ?assert(maps:get(score, lists:last(History)) > maps:get(score, hd(History))),
    %% A regression check on these three fixed validation scenarios, not a
    %% promise that any evolved network generalizes to arbitrary worlds.
    Food = fun(Brain) -> lists:sum([maps:get(eaten, life_world:evaluate(Brain, S))
                                    || S <- [101,202,303]]) end,
    ?assert(Food(Champion) > Food(Initial))
end}.

export_roundtrip_test() ->
    Dir = filename:join("/tmp", "life_export_" ++ integer_to_list(erlang:system_time(microsecond))),
    Path = filename:join(Dir, "index.html"),
    try
        ok = life_demo:export(Path, #{generations => 0, population => 4, verbose => false}),
        {ok, [Saved]} = file:consult(filename:join(Dir, "champion.term")),
        Brain = maps:get(champion, Saved),
        ?assertEqual(76, length(life_brain:weights(Brain))),
        ?assertEqual(life_world:evaluate(Brain, 101),
                     maps:without([frames], life_world:episode(Brain, 101))),
        {ok, Html} = file:read_file(Path),
        ?assertEqual(nomatch, binary:match(Html, <<"__LIFE_DATA__">>)),
        {ok, Json} = file:read_file(filename:join(Dir, "experiment.json")),
        ?assertNotEqual(nomatch, binary:match(Html, Json))
    after
        [file:delete(filename:join(Dir, Name)) || Name <- ["index.html", "experiment.json", "champion.term", "exoself.ets"]],
        file:del_dir(Dir)
    end.
