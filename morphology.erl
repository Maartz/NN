-module(morphology).
-compile(export_all).
-include("records.hrl").

get_InitSensor(Morphology) ->
    Sensors = morphology:Morphology(sensors),
    lists:nth(1, Sensors).

get_InitActuator(Morphology) ->
    Actuators = morphology:Morphology(actuators),
    lists:nth(1, Actuators).

get_Sensors(Morphology) ->
    morphology:Morphology(sensors).

get_Actuators(Morphology) ->
    morphology:Morphology(actuators).

xor_mimic(sensors) ->
    [#sensor{id = {sensor, helpers:generate_id()}, name = xor_GetInput, scape = {private, xor_sim}, vector_length = 2}];
xor_mimic(actuators) ->
    [#actuator{id = {actuator, helpers:generate_id()}, name = xor_SendOutput, scape = {private, xor_sim}, vector_length = 1}].