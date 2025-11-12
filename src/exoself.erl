%%% # ExoSelf Module
%%%
%%% ExoSelf is the **top-level orchestrator** that manages the complete lifecycle
%%% of a neural network agent. It creates the phenotype (running network) from
%%% the genotype (stored specification) and implements the training loop using
%%% perturbation-based learning.
%%%
%%% ## Responsibilities
%%%
%%% - **Phenotype Construction**: Spawn all processes (cortex, sensors, neurons, actuators, scapes)
%%% - **Process Linking**: Initialize all cerebral units with their connection configurations
%%% - **Training Loop**: Coordinate evaluation cycles with weight perturbations
%%% - **Learning Algorithm**: Implement hill-climbing optimization via random weight changes
%%% - **Genotype Backup**: Save trained weights back to persistent storage
%%% - **Process Cleanup**: Graceful termination of all spawned processes
%%%
%%% ## Process Architecture
%%%
%%% ExoSelf spawns and manages:
%%% - **1 Cortex**: Network coordinator
%%% - **N Sensors**: Input layer processes
%%% - **M Neurons**: Hidden and output layer processing units
%%% - **K Actuators**: Output layer processes
%%% - **S Scapes**: Environment simulators (private instances)
%%%
%%% All processes communicate via PIDs stored in an ETS table (`IdsNPIds`)
%%% that provides bidirectional ID ↔ PID mapping.
%%%
%%% ## Training Algorithm (Perturbation-Based Learning)
%%%
%%% A simple yet effective evolutionary approach without backpropagation:
%%%
%%% 1. **Evaluate**: Run network, collect fitness from cortex
%%% 2. **Compare**: Check if fitness improved
%%%    - **Improved**: Backup new weights, reset attempt counter
%%%    - **Degraded**: Restore previous weights, increment attempt counter
%%% 3. **Perturb**: Randomly select neurons (probability = 1/√N) and perturb their weights
%%% 4. **Repeat**: Continue until MAX_ATTEMPTS consecutive failures or fitness target reached
%%% 5. **Terminate**: Save final weights to genotype file
%%%
%%% ### Perturbation Probability
%%% ```
%%% P(perturb neuron) = 1 / sqrt(total_neurons)
%%% '''
%%% This ensures approximately √N neurons perturbed per iteration,
%%% balancing exploration (too many changes) vs exploitation (too few).
%%%
%%% ## Lifecycle Phases
%%%
%%% ```
%%% [map/1] → [prep/2] → [loop/13] → [backup & terminate]
%%%    ↓          ↓           ↓              ↓
%%%  Load     Spawn      Train         Save & Exit
%%% Genotype  Processes  Cycles        Results
%%% '''
%%%
%%% ## Message Protocol
%%%
%%% **Input Messages**:
%%% - `{CortexPId, evaluation_completed, Fitness, Cycles, Time}' - Evaluation results
%%% - `{NeuronPId, NId, WeightTuples}' - Weight backup from neuron
%%%
%%% **Output Messages**:
%%% - `{self(), weight_backup}' - Save current weights
%%% - `{self(), weight_restore}' - Restore previous weights
%%% - `{self(), weight_perturb}' - Randomly change weights
%%% - `{self(), reactivate}' - Start next evaluation cycle
%%% - `{self(), get_backup}' - Request neuron's current weights
%%% - `{self(), terminate}' - Shutdown process

-module(exoself).
-compile(export_all).
-include("records.hrl").

%% Maximum consecutive failed attempts before training terminates
-define(MAX_ATTEMPTS, 50).

-export([map/0, map/1]).

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Map genotype to phenotype (default filename: ffnn)
%%
%% Convenience function that loads the default genotype file
%% and spawns an ExoSelf process to train it.
%%
%% === Returns ===
%% PID of the spawned ExoSelf process
%%
%% === Examples ===
%% ```
%% exoself:map().
%% '''
-spec map() -> pid().
map() -> map(ffnn).

%% @doc Map genotype to phenotype
%%
%% Loads a neural network genotype from file and spawns an ExoSelf
%% process to construct the phenotype and begin training.
%%
%% This is the main entry point for running a neural network.
%%
%% === Parameters ===
%% - `FileName' - Name of the genotype file (atom)
%%%
%% === Returns ===
%% PID of the spawned ExoSelf process
%%
%% === Examples ===
%% ```
%% % Create a genotype first
%% genotype:construct(my_network, xor_mimic, [3]).
%%
%% % Map it to a running phenotype
%% exoself:map(my_network).
%% '''
-spec map(atom()) -> pid().
map(FileName) ->
    Genotype = genotype:load_from_file(FileName),
    spawn(exoself, prep, [FileName, Genotype]).

%%==============================================================================
%% Internal Functions - Phenotype Construction
%%==============================================================================

%% @private
%% Preparation phase - construct phenotype from genotype
%%
%% This function performs the complete phenotype instantiation:
%% 1. Create ETS table for ID/PID mapping
%% 2. Spawn all scapes (environment simulators)
%% 3. Spawn all cerebral units (cortex, sensors, neurons, actuators)
%% 4. Link processes by sending configuration messages
%% 5. Enter training loop
%%
%% The IdsNPIds ETS table maintains bidirectional mappings:
%% - `{Id, PId}' - Genotype ID to process PID
%% - `{PId, Id}' - Process PID to genotype ID
%%
%% This allows efficient lookups in both directions during linking.
prep(FileName, Genotype) ->
    % io:format("ExoSelf: Starting prep for ~p~n", [FileName]),
    rand:seed(exsplus),
    IdsNPIds = ets:new(idsNpids, [set, private]),
    Cx = genotype:read(Genotype, cortex),
    Sensor_Ids = Cx#cortex.sensor_ids,
    Actuator_Ids = Cx#cortex.actuator_ids,
    NIds = Cx#cortex.neuron_ids,
    ScapePIds = spawn_Scapes(IdsNPIds, Genotype, Sensor_Ids, Actuator_Ids),
    spawn_CerebralUnits(IdsNPIds, cortex, [Cx#cortex.id]),
    spawn_CerebralUnits(IdsNPIds, sensor, Sensor_Ids),
    spawn_CerebralUnits(IdsNPIds, actuator, Actuator_Ids),
    spawn_CerebralUnits(IdsNPIds, neuron, NIds),
    link_Sensors(Genotype, Sensor_Ids, IdsNPIds),
    link_Actuators(Genotype, Actuator_Ids, IdsNPIds),
    link_Neurons(Genotype, NIds, IdsNPIds),
    {SPIds, NPIds, APIds} = link_Cortex(Cx, IdsNPIds),
    Cx_PId = ets:lookup_element(IdsNPIds, Cx#cortex.id, 2),
    % io:format("ExoSelf: Entering main loop~n"),
    loop(FileName, Genotype, IdsNPIds, Cx_PId, SPIds, NPIds, APIds, ScapePIds, 0, 0, 0, 0, 1).

%%==============================================================================
%% Training Loop
%%==============================================================================

%% @private
%% Main training loop - perturbation-based learning
%%
%% Implements a simple hill-climbing algorithm:
%% - After each evaluation, compare fitness to previous best
%% - If improved: backup weights, reset attempt counter
%% - If degraded: restore previous weights, increment attempts
%% - Perturb random subset of neurons (P = 1/√N per neuron)
%% - Terminate after MAX_ATTEMPTS consecutive failures
%%
%% === Parameters ===
%% - `FileName' - Genotype file name for backup
%% - `Genotype' - ETS table reference
%% - `IdsNPIds' - ID/PID mapping table
%% - `Cx_PId' - Cortex process PID
%% - `SPIds' - Sensor process PIDs
%% - `NPIds' - Neuron process PIDs
%% - `APIds' - Actuator process PIDs
%% - `ScapePIds' - Scape process PIDs
%% - `HighestFitness' - Best fitness achieved so far
%% - `EvalAcc' - Total evaluations performed
%% - `CycleAcc' - Total cycles executed
%% - `TimeAcc' - Total time elapsed (microseconds)
%% - `Attempt' - Consecutive failed attempts counter
%%
%% === Training Termination ===
%% Training ends when `Attempt >= MAX_ATTEMPTS' (default: 50).
%% At termination:
%% 1. Backup final weights to genotype
%% 2. Terminate all processes
%% 3. Report statistics (fitness, evaluations, cycles, time)
%% 4. Notify trainer process if registered
loop(FileName, Genotype, IdsNPIds, Cx_PId, SPIds, NPIds, APIds, ScapePIds, HighestFitness, EvalAcc, CycleAcc, TimeAcc, Attempt) ->
    % io:format("ExoSelf: Waiting for evaluation (Attempt ~p)~n", [Attempt]),
    receive
        {Cx_PId, evaluation_completed, Fitness, Cycles, Time} ->
            {U_HighestFitness, U_Attempt} = case Fitness > HighestFitness of
                true ->
                    % Fitness improved - backup new weights
                    [NPId ! {self(), weight_backup} || NPId <- NPIds],
                    {Fitness, 0};
                false ->
                    % Fitness degraded - restore previous weights
                    Perturbed_NPIds = get(perturbed),
                    [NPId ! {self(), weight_restore} || NPId <- Perturbed_NPIds],
                    {HighestFitness, Attempt + 1}
            end,
            case U_Attempt >= ?MAX_ATTEMPTS of
                true ->
                    % Training complete - save and terminate
                    U_CycleAcc = CycleAcc + Cycles,
                    U_TimeAcc = TimeAcc + Time,
                    backup_genotype(FileName, IdsNPIds, Genotype, NPIds),
                    terminate_phenotype(Cx_PId, SPIds, NPIds, APIds, ScapePIds),
                    io:format("Cortex:~p finished training. Genotype has been backed up.~n Fitness:~p~n TotEvaluations:~p~n TotCycles:~p~n TimeAcc:~p~n",
                             [Cx_PId, U_HighestFitness, EvalAcc, U_CycleAcc, U_TimeAcc]),

                    case whereis(trainer) of
                        undefined ->
                            ok;
                        PId ->
                            PId ! {self(), U_HighestFitness, EvalAcc, U_CycleAcc, U_TimeAcc}
                    end;
                false ->
                    % Continue training - perturb and reactivate
                    Tot_Neurons = length(NPIds),
                    MP = 1 / math:sqrt(Tot_Neurons),
                    Perturb_NPIds = [NPId || NPId <- NPIds, rand:uniform() < MP],
                    put(perturbed, Perturb_NPIds),
                    [NPId ! {self(), weight_perturb} || NPId <- Perturb_NPIds],
                    Cx_PId ! {self(), reactivate},
                    loop(FileName, Genotype, IdsNPIds, Cx_PId, SPIds, NPIds, APIds, ScapePIds, U_HighestFitness, EvalAcc + 1, CycleAcc + Cycles, TimeAcc + Time, U_Attempt)
            end
    end.

%%==============================================================================
%% Process Spawning
%%==============================================================================

%% @private
%% Spawn cerebral unit processes
%%
%% Creates processes for cortex, sensors, neurons, or actuators.
%% Registers bidirectional ID/PID mappings in the ETS table.
%%
%% Calls the `gen/2' function of the appropriate module dynamically.
spawn_CerebralUnits(IdsNPIds, CerebralUnitType, [Id | Ids]) ->
    PId = CerebralUnitType:gen(self(), node()),
    ets:insert(IdsNPIds, {Id, PId}),
    ets:insert(IdsNPIds, {PId, Id}),
    spawn_CerebralUnits(IdsNPIds, CerebralUnitType, Ids);
spawn_CerebralUnits(_IdsNPIds, _CerebralUnitType, []) ->
    true.

%% @private
%% Spawn scape (environment) processes
%%
%% Creates private scape instances for the agent. Only spawns
%% unique scapes (removes duplicates from sensor and actuator lists).
%%
%% Registers scape names and PIDs in the ETS table for lookup
%% during sensor/actuator linking.
%%
%% === Returns ===
%% List of spawned scape PIDs
spawn_Scapes(IdsNPIds, Genotype, Sensor_Ids, Actuator_Ids) ->
    Sensor_Scapes = [(genotype:read(Genotype, Id))#sensor.scape || Id <- Sensor_Ids],
    Actuator_Scapes = [(genotype:read(Genotype, Id))#actuator.scape || Id <- Actuator_Ids],
    Unique_Scapes = Sensor_Scapes ++ (Actuator_Scapes -- Sensor_Scapes),
    SN_Tuples = [{scape:gen(self(), node()), ScapeName} || {private, ScapeName} <- Unique_Scapes],
    [ets:insert(IdsNPIds, {ScapeName, PId}) || {PId, ScapeName} <- SN_Tuples],
    [ets:insert(IdsNPIds, {PId, ScapeName}) || {PId, ScapeName} <- SN_Tuples],
    [PId ! {self(), ScapeName} || {PId, ScapeName} <- SN_Tuples],
    [PId || {PId, _ScapeName} <- SN_Tuples].

%%==============================================================================
%% Process Linking
%%==============================================================================

%% @private
%% Link sensor processes
%%
%% Sends initialization messages to each sensor with:
%% - Sensor ID
%% - Cortex PID
%% - Scape PID (for percept requests)
%% - Sensor function name
%% - Vector length
%% - Fanout neuron PIDs (where to send outputs)
link_Sensors(Genotype, [SId | Sensor_Ids], IdsNPIds) ->
    R = genotype:read(Genotype, SId),
    SPId = ets:lookup_element(IdsNPIds, SId, 2),
    Cx_PId = ets:lookup_element(IdsNPIds, R#sensor.cortex_id, 2),
    SName = R#sensor.name,
    Fanout_Ids = R#sensor.fanout_ids,
    Fanout_PIds = [ets:lookup_element(IdsNPIds, Id, 2) || Id <- Fanout_Ids],
    Scape = case R#sensor.scape of
        {private, ScapeName} ->
            ets:lookup_element(IdsNPIds, ScapeName, 2)
    end,
    SPId ! {self(), {SId, Cx_PId, Scape, SName, R#sensor.vector_length, Fanout_PIds}},
    link_Sensors(Genotype, Sensor_Ids, IdsNPIds);
link_Sensors(_Genotype, [], _IdsNPIds) ->
    ok.

%% @private
%% Link actuator processes
%%
%% Sends initialization messages to each actuator with:
%% - Actuator ID
%% - Cortex PID
%% - Scape PID (for action execution)
%% - Actuator function name
%% - Fanin neuron PIDs (where to receive inputs from)
link_Actuators(Genotype, [AId | Actuator_Ids], IdsNPIds) ->
    R = genotype:read(Genotype, AId),
    APId = ets:lookup_element(IdsNPIds, AId, 2),
    Cx_PId = ets:lookup_element(IdsNPIds, R#actuator.cortex_id, 2),
    AName = R#actuator.name,
    Fanin_Ids = R#actuator.fanin_ids,
    Fanin_PIds = [ets:lookup_element(IdsNPIds, Id, 2) || Id <- Fanin_Ids],
    Scape = case R#actuator.scape of
        {private, ScapeName} ->
            ets:lookup_element(IdsNPIds, ScapeName, 2)
    end,
    APId ! {self(), {AId, Cx_PId, Scape, AName, Fanin_PIds}},
    link_Actuators(Genotype, Actuator_Ids, IdsNPIds);
link_Actuators(_Genotype, [], _IdsNPIds) ->
    ok.

%% @private
%% Link neuron processes
%%
%% Sends initialization messages to each neuron with:
%% - Neuron ID
%% - Cortex PID
%% - Activation function name
%% - Input PIDs with weights: `[{PId, [W1, W2, ...]}, ..., Bias]'
%% - Output PIDs (where to send results)
%%
%% Converts genotype's ID-based weights to phenotype's PID-based weights.
link_Neurons(Genotype, [NId | Neuron_Ids], IdsNPIds) ->
    R = genotype:read(Genotype, NId),
    NPId = ets:lookup_element(IdsNPIds, NId, 2),
    Cx_PId = ets:lookup_element(IdsNPIds, R#neuron.cortex_id, 2),
    AFName = R#neuron.activation_function,
    Input_IdPs = R#neuron.input_ids,
    Output_Ids = R#neuron.output_ids,
    Input_PIdPs = convert_neuron_weights_to_process_weights(IdsNPIds, Input_IdPs, []),
    Output_PIds = [ets:lookup_element(IdsNPIds, Id, 2) || Id <- Output_Ids],
    NPId ! {self(), {NId, Cx_PId, AFName, Input_PIdPs, Output_PIds}},
    link_Neurons(Genotype, Neuron_Ids, IdsNPIds);
link_Neurons(_Genotype, [], _IdsNPIds) ->
    ok.

%% @private
%% Link cortex process
%%
%% Sends initialization message to cortex with all component PIDs:
%% - Cortex ID
%% - Sensor PIDs
%% - Neuron PIDs
%% - Actuator PIDs
%%
%% === Returns ===
%% Tuple `{SPIds, NPIds, APIds}' of all process ID lists
link_Cortex(Cx, IdsNPIds) ->
    Cx_Id = Cx#cortex.id,
    Cx_PId = ets:lookup_element(IdsNPIds, Cx_Id, 2),
    SIds = Cx#cortex.sensor_ids,
    AIds = Cx#cortex.actuator_ids,
    NIds = Cx#cortex.neuron_ids,
    SPIds = [ets:lookup_element(IdsNPIds, SId, 2) || SId <- SIds],
    NPIds = [ets:lookup_element(IdsNPIds, NId, 2) || NId <- NIds],
    APIds = [ets:lookup_element(IdsNPIds, AId, 2) || AId <- AIds],
    Cx_PId ! {self(), Cx_Id, SPIds, NPIds, APIds},
    {SPIds, NPIds, APIds}.

%%==============================================================================
%% Genotype Backup
%%==============================================================================

%% @private
%% Backup trained weights to genotype file
%%
%% After training completes, this function:
%% 1. Requests current weights from all neurons
%% 2. Converts PID-based weights back to ID-based weights
%% 3. Updates the genotype ETS table
%% 4. Saves genotype to disk
backup_genotype(FileName, IdsNPIds, Genotype, NPIds) ->
    Neuron_IdsNWeights = get_backup(NPIds, []),
    update_genotype(IdsNPIds, Genotype, Neuron_IdsNWeights),
    genotype:save_to_file(Genotype, FileName).
    % io:format("Finished updating genotype to file:~p~n", [FileName]).

%% @private
%% Request weight backups from all neurons
%%
%% Sends `{get_backup}' message to each neuron and collects
%% their responses: `{NPId, NId, WeightTuples}'.
get_backup([NPId | NPIds], Acc) ->
    NPId ! {self(), get_backup},
    receive
        {NPId, NId, WeightTuples} ->
            get_backup(NPIds, [{NId, WeightTuples} | Acc])
    end;
get_backup([], Acc) ->
    Acc.

%% @private
%% Update genotype with trained weights
%%
%% Converts PID-based weights back to ID-based format and
%% updates neuron records in the genotype table.
update_genotype(IdsNPIds, Genotype, [{N_Id, PIdPs} | WeightPs]) ->
    N = genotype:read(Genotype, N_Id),
    Updated_InputIdPs = convert_process_weights_to_neuron_weights(IdsNPIds, PIdPs, []),
    U_N = N#neuron{input_ids = Updated_InputIdPs},
    genotype:write(Genotype, U_N),
    update_genotype(IdsNPIds, Genotype, WeightPs);
update_genotype(_IdsNPIds, _Genotype, []) ->
    ok.

%%==============================================================================
%% Weight Conversion Utilities
%%==============================================================================

%% @private
%% Convert genotype's ID-based weights to phenotype's PID-based weights
%%
%% Transforms: `[{Id, Weights}, ..., {bias, Bias}]'
%% Into:       `[{PId, Weights}, ..., Bias]'
%%
%% Used during neuron linking to prepare runtime weight format.
convert_neuron_weights_to_process_weights(_IdsNPIds, [{bias, Bias}], Acc) ->
    lists:reverse([Bias | Acc]);
convert_neuron_weights_to_process_weights(IdsNPIds, [{Id, Weights} | Fanin_IdPs], Acc) ->
    convert_neuron_weights_to_process_weights(IdsNPIds, Fanin_IdPs,
                                             [{ets:lookup_element(IdsNPIds, Id, 2), Weights} | Acc]).

%% @private
%% Convert phenotype's PID-based weights back to genotype's ID-based weights
%%
%% Transforms: `[{PId, Weights}, ..., Bias]'
%% Into:       `[{Id, Weights}, ..., {bias, Bias}]'
%%
%% Used during genotype backup to save trained weights to disk.
convert_process_weights_to_neuron_weights(IdsNPIds, [{PId, Weights} | Input_PIdPs], Acc) ->
    convert_process_weights_to_neuron_weights(
        IdsNPIds,
        Input_PIdPs,
        [{ets:lookup_element(IdsNPIds, PId, 2), Weights} | Acc]
    );
convert_process_weights_to_neuron_weights(_IdsNPIds, [Bias], Acc) ->
    lists:reverse([{bias, Bias} | Acc]).

%%==============================================================================
%% Termination
%%==============================================================================

%% @private
%% Terminate all phenotype processes
%%
%% Sends terminate messages to all spawned processes in proper order:
%% 1. Sensors (stop sensing)
%% 2. Actuators (stop acting)
%% 3. Neurons (stop processing)
%% 4. Scapes (stop simulating)
%% 5. Cortex (stop coordinating)
terminate_phenotype(Cx_PId, SPIds, NPIds, APIds, ScapePIds) ->
    [PId ! {self(), terminate} || PId <- SPIds],
    [PId ! {self(), terminate} || PId <- APIds],
    [PId ! {self(), terminate} || PId <- NPIds],
    [PId ! {self(), terminate} || PId <- ScapePIds],
    Cx_PId ! {self(), terminate}.
