-module(constructor).

-export([construct_genotype/3, construct_genotype/4]).

-include("records.hrl").

construct_genotype(SensorName, ActuatorName, HiddenLayerDensities) ->
    construct_genotype(ffnn, SensorName, ActuatorName, HiddenLayerDensities).

construct_genotype(FileName, SensorName, ActuatorName, HiddenLayerDensities) ->
    S = create_sensor(SensorName),
    A = create_actuator(ActuatorName),
    Output_VL = A#actuator.vector_length,
    LayerDensities = lists:append(HiddenLayerDensities, [Output_VL]),
    Cx_id = {cortex_id, helpers:generate_id()},

    Neurons = create_neuro_layers(Cx_id, S, A, LayerDensities),
    [Input_Layer | _] = Neurons,
    [Output_Layer | _] = lists:reverse(Neurons),
    FL_NIds = [N#neuron.id || N <- Input_Layer],
    LL_NIds = [N#neuron.id || N <- Output_Layer],
    NIds = [N#neuron.id || N <- lists:flatten(Neurons)],
    Sensor = S#sensor{cortex_id = Cx_id, fanout_ids = FL_NIds},
    Actuator = A#actuator{cortex_id = Cx_id, fanin_ids = LL_NIds},
    Cortex = create_cortex(Cx_id, [S#sensor.id], [A#actuator.id], NIds),
    Genotype = lists:flatten([Cortex, Sensor, Actuator | Neurons]),
    {ok, File} = file:open(FileName, write),
    lists:foreach(fun(X) -> io:format(File, "~p.~n", [X]) end, Genotype),
    file:close(File).

%%%%%%%%%%%%%%%%%%%
%     Private     %
%%%%%%%%%%%%%%%%%%%

create_sensor(SensorName) ->
    case SensorName of
        rng ->
            #sensor{id = {sensor, helpers:generate_id()},
                    name = rng,
                    vector_length = 2};
        _ ->
            exit("System does not yet support a sensor by the name ~p", [SensorName])
    end.

create_actuator(ActuatorName) ->
    case ActuatorName of
        pts ->
            #actuator{id = {actuator, helpers:generate_id()},
                      name = pts,
                      vector_length = 1};
        _ ->
            exit("System does not yet support an actuator by the name ~p", [ActuatorName])
    end.

create_neuro_layers(Cortex_id, Sensor, Actuator, LayerDensities) ->
    Inputs_IdPs = [{Sensor#sensor.id, Sensor#sensor.vector_length}],
    Total_Layers = length(LayerDensities),
    [FL_Neurons | Next_LDs] = LayerDensities,
    Nlds = [{neuron, {1, Id}} || Id <- helpers:generate_ids(FL_Neurons, [])],
    create_neuro_layers(Cortex_id,
                        Actuator#actuator.id,
                        1,
                        Total_Layers,
                        Inputs_IdPs,
                        Nlds,
                        Next_LDs,
                        []).

create_neuro_layers(Cortex_id,
                    Actuator_id,
                    LayerIndex,
                    Total_Layers,
                    Input_IdPs,
                    NIds,
                    [Next_LD | LDs],
                    Acc) ->
    Output_Nlds = [{neuron, {LayerIndex + 1, Id}} || Id <- helpers:generate_ids(Next_LD, [])],
    Layer_Neurons = create_neuro_layer(Cortex_id, Input_IdPs, NIds, Output_Nlds, []),
    Next_Input_IdPs = [{NId, 1} || NId <- NIds],
    create_neuro_layers(Cortex_id,
                        Actuator_id,
                        LayerIndex + 1,
                        Total_Layers,
                        Next_Input_IdPs,
                        Output_Nlds,
                        LDs,
                        [Layer_Neurons | Acc]);
create_neuro_layers(Cortex_id,
                    Actuator_id,
                    Total_Layers,
                    Total_Layers,
                    Input_IdPs,
                    Nlds,
                    [],
                    Acc) ->
    Output_Ids = [Actuator_id],
    Layer_Neurons = create_neuro_layer(Cortex_id, Input_IdPs, Nlds, Output_Ids, []),
    lists:reverse([Layer_Neurons | Acc]).

create_neuro_layer(Cortex_id, Input_IdPs, [Id | Nlds], Output_Ids, Acc) ->
    Neuron = create_neuron(Input_IdPs, Id, Cortex_id, Output_Ids),
    create_neuro_layer(Cortex_id, Input_IdPs, Nlds, Output_Ids, [Neuron | Acc]);
create_neuro_layer(_, _, [], _, Acc) ->
    Acc.

create_neuron(Input_IdPs, Neuron_id, Cortex_id, Output_Ids) ->
    ProperInputIdPs = create_neural_input(Input_IdPs, []),
    #neuron{id = Neuron_id,
            cortex_id = Cortex_id,
            activation_function = tanh,
            input_ids = ProperInputIdPs,
            output_ids = Output_Ids}.

create_neural_input([{InputId, InputVL} | InputIdPs], Acc) ->
    Weights = create_neural_weights(InputVL, []),
    create_neural_input(InputIdPs, [{InputId, Weights} | Acc]);
create_neural_input([], Acc) ->
    lists:reverse([{bias, rand:uniform() - 0.5} | Acc]).

create_neural_weights(0, Acc) ->
    Acc;
create_neural_weights(Index, Acc) ->
    Weight = rand:uniform() - 0.5,
    create_neural_weights(Index - 1, [Weight | Acc]).

create_cortex(Cortex_id, Sensor_ids, Actuator_ids, Neuron_ids) ->
    #cortex{id = Cortex_id,
            sensor_ids = Sensor_ids,
            actuator_ids = Actuator_ids,
            neuron_ids = Neuron_ids}.
