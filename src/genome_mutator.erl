-module(genome_mutator).
-compile(export_all).
-include("records.hrl").
-define(MUTATORS, [
    mutate_weights,
    add_bias,
    mutate_activation_function,
    remove_bias,
    add_outlink,
    remove_outlink,
    add_inlink,
    remove_inlink,
    add_neuron,
    remove_neuron,
    outsplice,
    add_sensor,
    remove_sensor,
    add_actuator,
    remove_actuator
]).

-define(ACTUATORS, morphology:get_InitActuators(A#agent.morphology)).
-define(SENSORS, morphology:get_InitSensors(A#agent.morphology)).

test() ->
    Result = mutate(test),
    case Result of
        {atomic, _} -> io:format("Mutation successful~n");
        _ -> io:format("Mutation failed~n")
    end.

mutate(_Agent_ID) ->
    {aborted, not_implemented}.
