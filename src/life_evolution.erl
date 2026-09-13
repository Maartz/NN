%% Selection sees only the training seeds. World evaluation has its own RNG.
-module(life_evolution).
-export([run/1, evaluate_population/2]).

run(Options) ->
    Seed = maps:get(seed, Options, 42),
    Size = maps:get(population, Options, 32),
    Generations = maps:get(generations, Options, 40),
    true = is_integer(Size) andalso Size >= 4 andalso Size =< 256,
    true = is_integer(Generations) andalso Generations >= 0,
    rand:seed(exsplus, Seed),
    Seeds = [11,22,33],
    Initial = evaluate_population([life_brain:new() || _ <- lists:seq(1, Size)], Seeds),
    {Best, _} = hd(Initial),
    {Champion, History} = evolve(Initial, Seeds, 0, Generations, Size, [], Options),
    #{initial => Best, champion => Champion, history => History,
      training_seeds => Seeds, seed => Seed, population => Size, generations => Generations,
      evaluations => Size*(Generations+1)}.

evolve(Ranked, Seeds, Generation, Last, Size, History, Options) ->
    {Best, Metrics} = hd(Ranked),
    Entry = Metrics#{generation => Generation, evaluations => Size*(Generation+1)},
    case maps:get(verbose, Options, true) andalso Generation rem 5 =:= 0 of
        true -> io:format("Generation ~2w / ~w | score ~7.2f | food ~4.1f / 14~n",
                         [Generation, Last, maps:get(score, Metrics), maps:get(eaten, Metrics)]);
        false -> ok
    end,
    case Generation =:= Last of
        true -> {Best, lists:reverse([Entry | History])};
        false ->
            Elites = [Brain || {Brain, _} <- lists:sublist(Ranked, 4)],
            Children = [life_brain:mutate(lists:nth(rand:uniform(4), Elites))
                        || _ <- lists:seq(1, Size-4)],
            Next = evaluate_population(Elites ++ Children, Seeds),
            evolve(Next, Seeds, Generation+1, Last, Size, [Entry | History], Options)
    end.

%% One worker per candidate, ordinary function calls inside its small network.
%% Results are collected in population order, so scheduling cannot affect ties.
evaluate_population(Brains, Seeds) when length(Seeds) > 0 ->
    Owner = self(),
    Workers = [spawn_monitor(fun() ->
        try
            Episodes = [life_world:evaluate(Brain, Seed) || Seed <- Seeds],
            Average = fun(Key) -> lists:sum([maps:get(Key, E) || E <- Episodes])/length(Seeds) end,
            Owner ! {self(), result, #{score => Average(score), eaten => Average(eaten)}}
        catch Class:Reason -> Owner ! {self(), failed, {Class, Reason}}
        end
    end) || Brain <- Brains],
    try
        Metrics = [collect(Worker) || Worker <- Workers],
        Indexed = [{-maps:get(score, M), Index, {B,M}}
                   || {Index, {B,M}} <- lists:zip(lists:seq(1, length(Brains)), lists:zip(Brains, Metrics))],
        [Pair || {_, _, Pair} <- lists:sort(Indexed)]
    after
        [begin exit(Pid, kill), demonitor(Ref, [flush]),
               receive {Pid, _, _} -> ok after 0 -> ok end
         end || {Pid, Ref} <- Workers]
    end.

collect({Pid, Ref}) ->
    receive
        {Pid, result, Metrics} -> Metrics;
        {Pid, failed, Reason} -> error({evaluation_failed, Reason});
        {'DOWN', Ref, process, Pid, Reason} -> error({evaluation_failed, Reason})
    after 5000 -> error({evaluation_timeout, Pid})
    end.
