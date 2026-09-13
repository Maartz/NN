-module(life_exoself_tests).
-include_lib("eunit/include/eunit.hrl").

%% Capacity witness, not a learned solution. The horizontal score dominates
%% whenever dx is nonzero: 2*tanh(6/11) > tanh(1). Otherwise dy decides.
%% This reproduces scripted_action with the existing 7 -> 6 -> 4 architecture.
handwired_capacity_test_() -> {timeout, 15, fun() -> with_dir(fun(Dir) ->
    Zero = lists:duplicate(8, 0.0),
    Brain = #{hidden => [[6.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                         [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                         Zero,Zero,Zero,Zero],
              output => [[0.0,-1.0,0.0,0.0,0.0,0.0,0.0],
                         [2.0,0.0,0.0,0.0,0.0,0.0,0.0],
                         [0.0,1.0,0.0,0.0,0.0,0.0,0.0],
                         [-2.0,0.0,0.0,0.0,0.0,0.0,0.0]]},
    File = filename:join(Dir, "handwired"),
    ok = life_exoself:save(Brain, File),
    [begin
        Reference = life_world:episode(scripted, Seed),
        Actual = life_exoself:episode(File, Seed),
        ?assertEqual(14, maps:get(eaten, Actual)),
        ?assertEqual(cleared, maps:get(status, Actual)),
        ?assertEqual(maps:without([frames], Reference), maps:without([frames], Actual)),
        ?assertEqual([maps:get(action,F) || F <- maps:get(frames,Reference)],
                     [maps:get(action,F) || F <- maps:get(frames,Actual)])
    end || Seed <- lists:seq(2001,2100)]
end) end}.

actor_parity_test_() -> {timeout, 15, fun() -> with_dir(fun(Dir) ->
    [begin
        rand:seed(exsplus, Seed),
        Brain = life_brain:new(),
        File = filename:join(Dir, integer_to_list(Seed)),
        ok = life_exoself:save(Brain, File),
        ?assertEqual(Brain, life_exoself:load(File)),
        [assert_episode(life_world:episode(Brain, World), life_exoself:episode(File, World))
         || World <- [101,202,303]],
        ?assertEqual(Brain, life_exoself:load(File))
    end || Seed <- [42,7]]
end) end}.

training_budget_and_roundtrip_test_() -> {timeout, 15, fun() -> with_dir(fun(Dir) ->
    rand:seed(exsplus, 42),
    Brain = life_brain:new(),
    Runs = [begin
        File = filename:join(Dir, Name),
        ok = life_exoself:save(Brain, File),
        R = life_exoself:train(File, #{seed => 42, evaluation_limit => 32, max_attempts => 32}),
        Saved = life_exoself:load(File),
        History = maps:get(history, R),
        ?assertEqual(32, maps:get(evaluations, R)),
        ?assertEqual(lists:seq(1,32), [maps:get(evaluations,H) || H <- History]),
        Scores = [maps:get(score,H) || H <- History],
        ?assertEqual(lists:sort(Scores), Scores),
        ?assertEqual(mean_score(Brain), maps:get(score, hd(History))),
        ?assertEqual(mean_score(Saved), maps:get(score, R)),
        [assert_episode(life_world:episode(Saved,S), life_exoself:episode(File,S)) || S <- [101,202,303]],
        {Saved, History}
    end || Name <- ["first", "second"]],
    [First,Second] = Runs,
    ?assertEqual(First,Second)
end) end}.

morphology_entrypoint_test() -> with_dir(fun(Dir) ->
    File = filename:join(Dir, "constructed"),
    genotype:construct(File, life_mimic, [6], #{seed => 42}),
    Brain = life_exoself:load(File),
    R = life_exoself:train(File, #{evaluation_limit => 1}),
    ?assertEqual(mean_score(Brain), maps:get(score,R)),
    ExpectedCycles = lists:sum([maps:get(steps, life_world:evaluate(Brain,S)) || S <- [11,22,33]]),
    ?assertEqual(ExpectedCycles, maps:get(cycles,R))
end).

scape_failure_test() -> with_dir(fun(Dir) ->
    rand:seed(exsplus, 42),
    File = filename:join(Dir,"bad_scape"),
    ok = life_exoself:save(life_brain:new(), File),
    ?assertException(error, {exoself_failed, _},
        life_exoself:train(File, #{scape_options => #{life_sim => #{seeds => []}}}))
end).

matched_budget_test() -> with_dir(fun(Dir) ->
    Population = life_evolution:run(#{population => 8, generations => 2, verbose => false}),
    R = life_exoself:compare(Population, filename:join(Dir,"matched")),
    ?assertEqual(8, maps:get(initial_evaluations,R)),
    ?assertEqual(16, maps:get(evaluations,R)),
    ?assertEqual(maps:get(evaluations,Population), maps:get(total_evaluations,R)),
    ?assertEqual(72, maps:get(episode_evaluations,R)),
    ?assertEqual(8, maps:get(evaluations,hd(maps:get(history,R)))),
    ?assertEqual(24, maps:get(evaluations,lists:last(maps:get(history,R)))),
    ?assertEqual(mean_score(maps:get(champion,R)),maps:get(score,R))
end).

mean_score(Brain) ->
    lists:sum([maps:get(score,life_world:evaluate(Brain,S)) || S <- [11,22,33]])/3.

assert_episode(Expected, Actual) ->
    ?assertEqual(maps:without([frames],Expected), maps:without([frames],Actual)),
    EFrames = maps:get(frames,Expected),
    AFrames = maps:get(frames,Actual),
    ?assertEqual(length(EFrames), length(AFrames)),
    [begin
        ?assertEqual(maps:without([outputs],E), maps:without([outputs],A)),
        EO = maps:get(outputs,E), AO = maps:get(outputs,A),
        ?assertEqual(length(EO),length(AO)),
        [?assert(abs(X-Y)<1.0e-12) || {X,Y} <- lists:zip(EO,AO)]
    end || {E,A} <- lists:zip(EFrames,AFrames)].

with_dir(Fun) ->
    Dir = filename:join("/tmp", "life_actors_" ++ integer_to_list(erlang:system_time(microsecond))),
    ok = file:make_dir(Dir),
    try Fun(Dir)
    after
        [file:delete(F) || F <- filelib:wildcard(filename:join(Dir,"*"))],
        file:del_dir(Dir)
    end.
