%% Bridge the small demo's weight representation to the book's real actors.
-module(life_exoself).
-include("records.hrl").
-export([save/2, load/1, train/2, episode/2, compare/2]).

save(#{hidden := Hidden, output := Output}, File) when length(Hidden) =:= 6, length(Output) =:= 4 ->
    [Sensor0] = morphology:life_mimic(sensors),
    [Actuator0] = morphology:life_mimic(actuators),
    SId = Sensor0#sensor.id,
    AId = Actuator0#actuator.id,
    HiddenIds = [{neuron,{1,I}} || I <- lists:seq(1,6)],
    OutputIds = [{neuron,{2,I}} || I <- lists:seq(1,4)],
    Sensor = Sensor0#sensor{cortex_id = cortex, fanout_ids = HiddenIds},
    Actuator = Actuator0#actuator{cortex_id = cortex, fanin_ids = OutputIds},
    HiddenNeurons = [begin
        {Weights,[Bias]} = lists:split(7, Row),
        #neuron{id = Id, cortex_id = cortex, activation_function = tanh,
                input_ids = [{SId,Weights},{bias,Bias}], output_ids = OutputIds}
    end || {Id,Row} <- lists:zip(HiddenIds,Hidden)],
    OutputNeurons = [begin
        {Weights,[Bias]} = lists:split(6, Row),
        #neuron{id = Id, cortex_id = cortex, activation_function = tanh,
                input_ids = [{HId,[W]} || {HId,W} <- lists:zip(HiddenIds,Weights)] ++ [{bias,Bias}],
                output_ids = [AId]}
    end || {Id,Row} <- lists:zip(OutputIds,Output)],
    Cortex = #cortex{id = cortex, sensor_ids = [SId], actuator_ids = [AId],
                     neuron_ids = HiddenIds ++ OutputIds},
    genotype:save_genotype(File, [Cortex,Sensor,Actuator] ++ HiddenNeurons ++ OutputNeurons).

load(File) ->
    Tab = genotype:load_from_file(File),
    try
        Cx = genotype:read(Tab, cortex),
        [AId] = Cx#cortex.actuator_ids,
        Actuator = genotype:read(Tab, AId),
        [FirstOutput | _] = OutputIds = Actuator#actuator.fanin_ids,
        First = genotype:read(Tab, FirstOutput),
        HiddenIds = [Id || {Id,_} <- First#neuron.input_ids, Id =/= bias],
        Hidden = [begin
            #neuron{activation_function = tanh, input_ids = [{_,Ws},{bias,B}]} = genotype:read(Tab, Id),
            Ws ++ [B]
        end || Id <- HiddenIds],
        Output = [begin
            N = genotype:read(Tab, Id),
            tanh = N#neuron.activation_function,
            [begin {_,[W]} = lists:keyfind(HId,1,N#neuron.input_ids), W end || HId <- HiddenIds]
                ++ [element(2,lists:keyfind(bias,1,N#neuron.input_ids))]
        end || Id <- OutputIds],
        6 = length(Hidden), 4 = length(Output),
        #{hidden => Hidden, output => Output}
    after ets:delete(Tab)
    end.

%% ExoSelf retains its own backup/restore/perturb algorithm. Options choose limits.
%% MODE D'EMPLOI : voir docs/TRAIN.md pour des commandes copiables hors ligne.
%% train/2 attend le résultat et écrit les meilleurs poids dans File.
%% Rappeler train/2 sur ce fichier poursuit depuis les poids sauvegardés.
%% Pour un départ indépendant, créer un NOUVEAU génotype avec construct/4.
%% Changer seulement Options.seed ne réinitialise pas les poids du fichier.
train(File, Options) ->
    Pid = exoself:map(File, Options#{progress_to => self()}),
    Ref = monitor(process, Pid),
    try await(Pid, Ref, [])
    after exit(Pid, kill), demonitor(Ref, [flush])
    end.

await(Pid, Ref, History) ->
    receive
        {Pid, progress, Entry} -> await(Pid, Ref, [Entry | History]);
        {Pid, Fitness, Evals, Cycles, Time} ->
            #{score => Fitness, evaluations => Evals, cycles => Cycles,
              microseconds => Time, history => lists:reverse(History)};
        {Pid, error, Reason} -> error({exoself_failed, Reason});
        {'DOWN', Ref, process, Pid, Reason} -> error({exoself_failed, Reason})
    after 10000 -> error(exoself_response_timeout)
    end.

%% Record decisions made by neuron actors from the saved genotype. One evaluation
%% performs no mutation. This also checks that replay fitness matches the engine.
%% Seed désigne ici un MONDE, pas une graine de perturbation. Utiliser des
%% mondes hors entraînement pour mesurer la généralisation. Ne pas sélectionner
%% le gagnant des redémarrages d'après ces épisodes : choisir sur le score train.
episode(File, Seed) ->
    Trace = make_ref(),
    Result = train(File, #{evaluation_limit => 1,
                         scape_options => #{life_sim => #{seeds => [Seed], trace_to => self(), trace_ref => Trace}}}),
    receive
        {life_trace, Trace, Seed, Episode} ->
            true = abs(maps:get(score,Episode) - maps:get(score,Result)) < 1.0e-9,
            maps:get(steps,Episode) =:= maps:get(cycles,Result) orelse error(replay_cycle_mismatch),
            Episode
    after 1000 -> error(missing_life_trace)
    end.


%% Charge both methods for the same initial selection. Remaining evaluations
%% use the original ExoSelf hill climber; validation episodes are outside training.
%% Cette fonction compare ExoSelf à life_evolution (la démo HTML existante).
%% Pour 1 x 10 000 contre 10 x 1 000, suivre l'atelier docs/TRAIN.md : c'est
%% une expérience différente, entre deux budgets de redémarrage d'ExoSelf.
compare(Experiment, File) ->
    Initial = maps:get(initial, Experiment),
    InitialCost = maps:get(population, Experiment),
    Total = maps:get(evaluations, Experiment),
    Remaining = Total - InitialCost,
    InitialScore = maps:get(score, hd(maps:get(history, Experiment))),
    ok = save(Initial, File),
    Start = #{evaluations => InitialCost, score => InitialScore},
    Result = case Remaining of
        0 -> #{score => InitialScore, evaluations => 0, cycles => 0,
               microseconds => 0, history => []};
        _ -> train(File, #{seed => maps:get(seed, Experiment),
                           evaluation_limit => Remaining, max_attempts => Remaining})
    end,
    Remaining = maps:get(evaluations, Result),
    History = [Start | [H#{evaluations => InitialCost+maps:get(evaluations,H)}
                       || H <- maps:get(history,Result)]],
    Result#{history => History, initial_evaluations => InitialCost,
            total_evaluations => Total, episode_evaluations => Total*3,
            champion => load(File)}.
