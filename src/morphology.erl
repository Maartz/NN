%% @doc Morphology Definitions for Neural Networks
%%
%% This module defines different neural network morphologies - configurations
%% of sensors and actuators for specific problem domains. Each morphology
%% specifies the input/output interface for a particular task.
%%
%% A morphology defines:
%% <ul>
%%   <li>Sensors: Input components that receive data from the environment</li>
%%   <li>Actuators: Output components that send actions to the environment</li>
%%   <li>Scapes: Private or public environment simulators</li>
%% </ul>
%%
%% Currently implemented morphologies:
%% <ul>
%%   <li>`xor_mimic' - XOR problem with 2 inputs and 1 output</li>
%% </ul>
%%
%% @author Neural Network Project
%% @version 1.0

-module(morphology).
-compile(export_all).
-include("records.hrl").

-export([
    get_InitSensor/1,
    get_InitSensors/1,
    get_InitActuator/1,
    get_InitActuators/1,
    get_Sensors/1,
    get_Actuators/1
]).

%%==============================================================================
%% API Functions - Sensor Retrieval
%%==============================================================================

%% @doc Get the first sensor for a morphology
%%
%% Retrieves the first sensor from the list of sensors defined for the
%% specified morphology. Useful when only one sensor is needed.
%%
%% @param Morphology The morphology type (e.g., xor_mimic)
%% @returns First sensor record from the morphology
%%
%% @see get_InitSensors/1
-spec get_InitSensor(Morphology :: morphology()) -> #sensor{}.
get_InitSensor(Morphology) ->
    Sensors = morphology:Morphology(sensors),
    lists:nth(1, Sensors).

%% @doc Get all sensors for a morphology
%%
%% Retrieves the complete list of sensors defined for the specified
%% morphology. Each sensor is pre-configured with its scape, vector length,
%% and function name.
%%
%% @param Morphology The morphology type (e.g., xor_mimic)
%% @returns List of sensor records
%%
%% @see get_InitSensor/1
-spec get_InitSensors(Morphology :: morphology()) -> [#sensor{}].
get_InitSensors(Morphology) ->
    morphology:Morphology(sensors).

%% @doc Alias for get_InitSensors/1
%%
%% @see get_InitSensors/1
-spec get_Sensors(Morphology :: morphology()) -> [#sensor{}].
get_Sensors(Morphology) ->
    morphology:Morphology(sensors).

%%==============================================================================
%% API Functions - Actuator Retrieval
%%==============================================================================

%% @doc Get the first actuator for a morphology
%%
%% Retrieves the first actuator from the list of actuators defined for the
%% specified morphology. Useful when only one actuator is needed.
%%
%% @param Morphology The morphology type (e.g., xor_mimic)
%% @returns First actuator record from the morphology
%%
%% @see get_InitActuators/1
-spec get_InitActuator(Morphology :: morphology()) -> #actuator{}.
get_InitActuator(Morphology) ->
    Actuators = morphology:Morphology(actuators),
    lists:nth(1, Actuators).

%% @doc Get all actuators for a morphology
%%
%% Retrieves the complete list of actuators defined for the specified
%% morphology. Each actuator is pre-configured with its scape, vector length,
%% and function name.
%%
%% @param Morphology The morphology type (e.g., xor_mimic)
%% @returns List of actuator records
%%
%% @see get_InitActuator/1
-spec get_InitActuators(Morphology :: morphology()) -> [#actuator{}].
get_InitActuators(Morphology) ->
    morphology:Morphology(actuators).

%% @doc Alias for get_InitActuators/1
%%
%% @see get_InitActuators/1
-spec get_Actuators(Morphology :: morphology()) -> [#actuator{}].
get_Actuators(Morphology) ->
    morphology:Morphology(actuators).

%%==============================================================================
%% Morphology Definitions
%%==============================================================================

%% @doc XOR Mimic Morphology
%%
%% Defines a morphology for solving the XOR (exclusive OR) problem.
%%
%% <strong>Problem Description:</strong>
%% XOR is a classic non-linearly separable problem that requires at least
%% one hidden layer to solve. The truth table is:
%% ```
%% Input1  Input2  Output
%%   -1      -1      -1
%%    1      -1       1
%%   -1       1       1
%%    1       1      -1
%% '''
%%
%% <strong>Sensors:</strong>
%% <ul>
%%   <li>Name: `xor_GetInput'</li>
%%   <li>Vector Length: 2 (two binary inputs)</li>
%%   <li>Scape: `{private, xor_sim}' - each agent gets its own simulator</li>
%% </ul>
%%
%% <strong>Actuators:</strong>
%% <ul>
%%   <li>Name: `xor_SendOutput'</li>
%%   <li>Vector Length: 1 (single binary output)</li>
%%   <li>Scape: `{private, xor_sim}' - same simulator as sensor</li>
%% </ul>
%%
%% @param Type Either 'sensors' or 'actuators'
%% @returns List containing one sensor or one actuator
%%
%% Example:
%% ```
%% 1> morphology:xor_mimic(sensors).
%% [#sensor{name=xor_GetInput, vector_length=2, ...}]
%% '''
-spec xor_mimic(Type :: sensors | actuators) -> [#sensor{}] | [#actuator{}].
xor_mimic(sensors) ->
    [#sensor{
        id = {sensor, helpers:generate_id()},
        name = xor_GetInput,
        scape = {private, xor_sim},
        vector_length = 2
    }];
xor_mimic(actuators) ->
    [#actuator{
        id = {actuator, helpers:generate_id()},
        name = xor_SendOutput,
        scape = {private, xor_sim},
        vector_length = 1
    }].
