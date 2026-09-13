-module(nn_tests).
-include_lib("eunit/include/eunit.hrl").
-include("records.hrl").

xor_roundtrip_test_() ->
    {timeout, 30, fun() -> with_dir(fun(Dir) ->
        Options = #{seed => 42, max_attempts => 500, evaluation_limit => 20000,
                    fitness_target => 20, timeout => 1000},
        Runs = [begin
            File = filename:join(Dir, Name),
            genotype:construct(File, xor_mimic, [3], #{seed => 42}),
            {Fitness, Evals, Cycles, _} = await(exoself:map(File, Options)),
            Records = load_records(File),
            Outputs = outputs(Records),
            io:format("XOR fitness=~p evals=~p outputs=~p~n", [Fitness, Evals, Outputs]),
            ?assertEqual(4 * Evals, Cycles),
            ?assert(Fitness >= 20),
            ?assert(lists:all(fun({Actual, Expected}) -> Actual * Expected > 0.9 end,
                             lists:zip(Outputs, [-1, 1, 1, -1]))),
            ?assert(abs(fitness(Outputs) - Fitness) < 1.0e-9),
            %% Run the persisted network through actors again, without perturbation.
            {ReloadFitness, 1, 4, _} = await(exoself:map(File, #{evaluation_limit => 1})),
            ?assert(abs(ReloadFitness - Fitness) < 1.0e-9),
            {Fitness, Evals, Outputs}
        end || Name <- ["first", "second"]],
        [First, Second] = Runs,
        ?assertEqual(First, Second),
        assert_no_components()
    end) end}.

trainer_delivery_test_() ->
    {timeout, 10, fun() -> with_dir(fun(Dir) ->
        %% Two concurrent trainers must each receive their own ExoSelf result.
        Pids = [trainer:go(xor_mimic, [2], 1, 12, inf,
                          #{seed => Seed, output_dir => Dir, max_attempts => 2})
                || Seed <- [11, 22]],
        [begin
            {F, E, C, _} = await(Pid),
            ?assert(F > 0), ?assert(E > 0), ?assert(E =< 12),
            ?assertEqual(4 * E, C)
         end || Pid <- Pids],
        BestFiles = filelib:wildcard(filename:join(Dir, "best_*")),
        ?assertEqual(2, length(BestFiles)),
        ?assertEqual([], filelib:wildcard(filename:join(Dir, "experimental_*"))),
        [load_records(File) || File <- BestFiles],
        assert_no_components()
    end) end}.

component_failure_test() -> with_dir(fun(Dir) ->
    File = filename:join(Dir, "crash"),
    Records = genotype:construct(File, xor_mimic, [2], #{seed => 1}),
    [Neuron | _] = [N || N = #neuron{} <- Records],
    Bad = Neuron#neuron{activation_function = missing_activation},
    ok = genotype:save_genotype(File, lists:keyreplace(Neuron#neuron.id, 2, Records, Bad)),
    ?assertMatch({error, {component_failed, _, _}}, await_error(exoself:map(File))),
    assert_no_components()
end).

evaluation_timeout_test() -> with_dir(fun(Dir) ->
    File = filename:join(Dir, "stalled"),
    Records = genotype:construct(File, xor_mimic, [2], #{seed => 1}),
    [Neuron | _] = [N || N = #neuron{} <- Records],
    %% A self-dependent input cannot produce the first forward message.
    Bad = Neuron#neuron{input_ids = [{Neuron#neuron.id, [1.0]}, {bias, 0.0}]},
    ok = genotype:save_genotype(File, lists:keyreplace(Neuron#neuron.id, 2, Records, Bad)),
    ?assertEqual({error, evaluation_timeout}, await_error(exoself:map(File, #{timeout => 20}))),
    assert_no_components()
end).

owner_death_test() -> with_dir(fun(Dir) ->
    File = filename:join(Dir, "owner_death"),
    Records = genotype:construct(File, xor_mimic, [2], #{seed => 1}),
    [Neuron | _] = [N || N = #neuron{} <- Records],
    Bad = Neuron#neuron{input_ids = [{Neuron#neuron.id, [1.0]}, {bias, 0.0}]},
    ok = genotype:save_genotype(File, lists:keyreplace(Neuron#neuron.id, 2, Records, Bad)),
    Test = self(),
    Owner = spawn(fun() ->
        Agent = exoself:map(File),
        Test ! {agent, Agent},
        receive stop -> ok end
    end),
    Agent = receive {agent, A} -> A after 1000 -> error(no_agent) end,
    Ref = monitor(process, Agent),
    exit(Owner, kill),
    receive
        {'DOWN', Ref, process, Agent, {{owner_down, Reason}, _}}
                when Reason =:= killed; Reason =:= noproc -> ok
    after 2000 -> exit(Agent, kill), error(orphaned_agent)
    end,
    assert_no_components()
end).

save_failure_test() -> with_dir(fun(Dir) ->
    ?assertException(error, {badmatch, {error, _}},
                     genotype:construct(filename:join([Dir, "missing", "file"]), xor_mimic, [2]))
end).

mutation_placeholder_test() ->
    ?assertEqual({aborted, not_implemented}, genome_mutator:mutate(test)).

await(Pid) ->
    Ref = monitor(process, Pid),
    Result = receive
        {Pid, F, E, C, T} -> {F, E, C, T};
        {Pid, error, Reason} -> error({unexpected_failure, Reason});
        {'DOWN', Ref, process, Pid, Reason} -> error({no_result, Reason})
    after 25000 -> exit(Pid, kill), error(result_timeout)
    end,
    receive {'DOWN', Ref, process, Pid, ExitReason} when ExitReason =:= normal; ExitReason =:= noproc -> ok
    after 1000 -> error(worker_did_not_stop)
    end,
    Result.

await_error(Pid) ->
    Ref = monitor(process, Pid),
    Reason = receive {Pid, error, R} -> R
    after 2000 -> exit(Pid, kill), error(missing_error)
    end,
    receive {'DOWN', Ref, process, Pid, ExitReason} -> ?assertNotEqual(normal, ExitReason)
    after 1000 -> error(worker_did_not_stop)
    end,
    Reason.

load_records(File) ->
    Tab = genotype:load_from_file(File),
    try ets:tab2list(Tab) after ets:delete(Tab) end.

outputs(Records) ->
    Cx = lists:keyfind(cortex, 2, Records),
    [Sensor] = Cx#cortex.sensor_ids,
    [ActuatorId] = Cx#cortex.actuator_ids,
    Actuator = lists:keyfind(ActuatorId, 2, Records),
    [begin
        Values = lists:foldl(fun(Id, Acc) ->
            N = lists:keyfind(Id, 2, Records),
            Sum = lists:sum([case Pair of
                {bias, Bias} -> Bias;
                {InputId, Weights} -> lists:sum([X * W || {X, W} <- lists:zip(maps:get(InputId, Acc), Weights)])
            end || Pair <- N#neuron.input_ids]),
            Acc#{Id => [math:tanh(Sum)]}
        end, #{Sensor => Input}, Cx#cortex.neuron_ids),
        [Output] = lists:append([maps:get(Id, Values) || Id <- Actuator#actuator.fanin_ids]),
        Output
    end || Input <- [[-1,-1], [1,-1], [-1,1], [1,1]]].

fitness(Outputs) ->
    1 / (math:sqrt(lists:sum([abs(A-B) || {A,B} <- lists:zip(Outputs, [-1,1,1,-1])])) + 0.00001).

assert_no_components() -> assert_no_components(100).
assert_no_components(Attempts) ->
    Modules = [neuron, sensor, actuator, cortex, scape],
    Live = [Pid || Pid <- processes(),
                   {initial_call, {M, _, _}} <- [process_info(Pid, initial_call)],
                   lists:member(M, Modules)],
    case {Live, Attempts} of
        {[], _} -> ok;
        {_, 0} -> ?assertEqual([], Live);
        _ -> timer:sleep(5), assert_no_components(Attempts - 1)
    end.

with_dir(Fun) ->
    Dir = filename:join("/tmp", "nn_test_" ++ integer_to_list(erlang:unique_integer([positive, monotonic]))),
    ok = file:make_dir(Dir),
    try Fun(Dir)
    after
        [file:delete(File) || File <- filelib:wildcard(filename:join(Dir, "*"))],
        file:del_dir(Dir)
    end.
