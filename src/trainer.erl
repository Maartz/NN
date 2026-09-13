-module(trainer).
-export([go/2, go/5, go/6]).
-define(MAX_ATTEMPTS,5).
-define(EVAL_LIMIT,inf).
-define(FITNESS_TARGET,inf).

go(Morphology, HiddenLayerDensities) ->
    go(Morphology, HiddenLayerDensities, ?MAX_ATTEMPTS, ?EVAL_LIMIT, ?FITNESS_TARGET).

go(Morphology, HiddenLayerDensities, MaxAttempts, EvalLimit, FitnessTarget) ->
    go(Morphology, HiddenLayerDensities, MaxAttempts, EvalLimit, FitnessTarget, #{}).

%% Options are passed to ExoSelf; seed controls construction and training.
%% output_dir selects where experimental and best genotype files are written.
go(Morphology, Layers, MaxAttempts, EvalLimit, Target, Options)
        when is_integer(MaxAttempts), MaxAttempts > 0,
             (EvalLimit =:= inf orelse (is_integer(EvalLimit) andalso EvalLimit > 0)) ->
    Caller = self(),
    spawn(fun() ->
        OwnerRef = monitor(process, Caller),
        case maps:find(seed, Options) of
            {ok, Seed} -> rand:seed(exsplus, Seed);
            error -> rand:seed(exsplus)
        end,
        Suffix = integer_to_list(erlang:system_time(microsecond)) ++ "_" ++
                 integer_to_list(erlang:unique_integer([positive, monotonic])),
        Dir = maps:get(output_dir, Options, "."),
        Exp = filename:join(Dir, "experimental_" ++ Suffix),
        Best = filename:join(Dir, "best_" ++ Suffix),
        try
            loop(Caller, OwnerRef, Morphology, Layers, MaxAttempts, EvalLimit,
                 Target, Options, Exp, Best, 0, -1, 0, 0, 0)
        catch
            Class:Reason:Stack ->
                Caller ! {self(), error, {Class, Reason}},
                erlang:raise(Class, Reason, Stack)
        after
            file:delete(Exp)
        end
    end).

loop(Caller, _OwnerRef, _Morphology, _Layers, MaxAttempts, EvalLimit, Target,
     _Options, _Exp, Best, Attempts, Fitness, Evals, Cycles, Time)
        when Attempts >= MaxAttempts; Evals >= EvalLimit; Fitness >= Target ->
    io:format("Best genotype: ~ts Fitness: ~p Evaluations: ~p~n", [Best, Fitness, Evals]),
    Caller ! {self(), Fitness, Evals, Cycles, Time};
loop(Caller, OwnerRef, Morphology, Layers, MaxAttempts, EvalLimit, Target,
     Options, Exp, Best, Attempts, BestFitness, Evals, Cycles, Time) ->
    genotype:construct(Exp, Morphology, Layers, #{seed => rand:uniform(1 bsl 58)}),
    Remaining = case EvalLimit of inf -> 10000; _ -> EvalLimit - Evals end,
    RunOptions = Options#{seed => rand:uniform(1 bsl 58), fitness_target => Target,
                          evaluation_limit => min(Remaining, maps:get(evaluation_limit, Options, 10000))},
    Agent = exoself:map(Exp, RunOptions),
    Ref = monitor(process, Agent),
    receive
        {Agent, Fitness, RunEvals, RunCycles, RunTime} ->
            demonitor(Ref, [flush]),
            {NextAttempts, NextFitness} = case Fitness > BestFitness of
                true ->
                    ok = file:rename(Exp, Best),
                    {0, Fitness};
                false -> {Attempts + 1, BestFitness}
            end,
            loop(Caller, OwnerRef, Morphology, Layers, MaxAttempts, EvalLimit, Target,
                 Options, Exp, Best, NextAttempts, NextFitness,
                 Evals + RunEvals, Cycles + RunCycles, Time + RunTime);
        {Agent, error, Reason} -> error({training_failed, Reason});
        {'DOWN', Ref, process, Agent, Reason} -> error({training_failed, Reason});
        {'DOWN', OwnerRef, process, Caller, Reason} -> error({owner_down, Reason});
        terminate -> exit(shutdown)
    end.
