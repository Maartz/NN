%%% # Cortex Module
%%%
%%% The cortex is the **network coordinator** that orchestrates the synchronous
%%% execution of the neural network. It runs as an independent process that
%%% manages the sensor-neuron-actuator evaluation cycles.
%%%
%%% ## Responsibilities
%%%
%%% - **Cycle Coordination**: Trigger sensors to start each evaluation cycle
%%% - **Synchronization**: Collect sync messages from all actuators
%%% - **Fitness Aggregation**: Accumulate fitness scores across cycles
%%% - **State Management**: Switch between active (running) and inactive (waiting) states
%%% - **Timing**: Track cycle count and execution time for performance metrics
%%% - **Termination**: Broadcast terminate messages to all cerebral units
%%%
%%% ## Process Lifecycle
%%%
%%% 1. **Spawning**: Created by ExoSelf via `gen/2'
%%% 2. **Initialization**: Receives PIDs of all sensors, neurons, and actuators
%%% 3. **Active Loop**: Triggers sensors, waits for actuator sync messages
%%% 4. **Evaluation Complete**: Sends results to ExoSelf, enters inactive state
%%% 5. **Reactivation**: ExoSelf sends reactivate message to start next evaluation
%%% 6. **Termination**: Broadcasts terminate to all units and exits
%%%
%%% ## Message Protocol
%%%
%%% **Input Messages**:
%%% - `{ExoSelf_PId, Id, SPIds, NPIds, APIds}' - Initialization with all PIDs
%%% - `{ActuatorPId, sync, Fitness, EndFlag}' - Actuator finished cycle
%%% - `{ExoSelf_PId, reactivate}' - Start new evaluation (from inactive state)
%%% - `{ExoSelf_PId, terminate}' - Shutdown cortex and all units
%%%
%%% **Output Messages**:
%%% - `{self(), sync}' - Trigger sensors to sense
%%% - `{self(), evaluation_completed, Fitness, Cycles, Time}' - Report to ExoSelf
%%% - `{self(), terminate}' - Shutdown signal to all units
%%%
%%% ## State Machine
%%%
%%% ```
%%% [Prep] -> {initialization} -> [Active]
%%%   ^                              |
%%%   |                              v
%%%   |                        {sync from actuators}
%%%   |                              |
%%%   |                              v
%%%   |                        {all actuators synced?}
%%%   |                              |
%%%   |                    No /------+------\ Yes, EndFlag > 0
%%%   |                      /               \
%%%   |                [Next Cycle]      [Inactive]
%%%   |                     |                |
%%%   +-----{reactivate}----+----------------+
%%% '''
%%%
%%% ## Evaluation Cycle Flow
%%%
%%% 1. Cortex sends `{sync}' to all sensors
%%% 2. Sensors request percepts from scapes, forward to neurons
%%% 3. Neurons compute and forward outputs through network layers
%%% 4. Actuators collect outputs, send to scapes, get fitness
%%% 5. Actuators send `{sync, Fitness, EndFlag}' to cortex
%%% 6. Cortex waits for ALL actuators to sync
%%% 7. If EndFlag > 0: Report to ExoSelf and go inactive
%%% 8. If EndFlag = 0: Start next cycle (goto step 1)

-module(cortex).
-compile(export_all).
-include("records.hrl").

%% Internal state record (not persisted)
-record(state, {
    id,
    exoself_pid,
    spids,
    npids,
    apids,
    cycle_acc = 0,
    fitness_acc = 0,
    endflag = 0,
    status
}).

-export([gen/2]).

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Spawn a new cortex process
%%
%% Creates a cortex process on the specified node. The cortex enters
%% a preparation state waiting for initialization with all unit PIDs.
%%
%% === Parameters ===
%% - `ExoSelf_PId' - PID of the ExoSelf orchestrator
%% - `Node' - Node where the cortex should be spawned
%%
%% === Returns ===
%% PID of the spawned cortex process
%%
%% === Examples ===
%% ```
%% CortexPId = cortex:gen(ExoSelfPId, node()).
%% '''
-spec gen(pid(), node()) -> pid().
gen(ExoSelf_PId, Node) ->
    spawn(Node, ?MODULE, prep, [ExoSelf_PId]).

%%==============================================================================
%% Internal Functions - Processing Loop
%%==============================================================================

%% @private
%% Preparation loop - wait for initialization
%%
%% Receives PIDs of all neural network components and starts
%% the first evaluation cycle by triggering all sensors.
prep(ExoSelf_PId) ->
    rand:seed(exsplus, os:timestamp()),
    receive
        {ExoSelf_PId, Id, SPIds, NPIds, APIds} ->
            % io:format("Cortex: Starting with ~p sensors, ~p neurons, ~p actuators~n",
            %          [length(SPIds), length(NPIds), length(APIds)]),
            put(start_time, os:timestamp()),
            [SPId ! {self(), sync} || SPId <- SPIds],
            loop(Id, ExoSelf_PId, SPIds, {APIds, APIds}, NPIds, 1, 0, 0, active)
    end.

%% @private
%% Active loop - waiting for actuator sync messages
%%
%% Collects sync messages from all actuators. Each actuator sends
%% a fitness score and end flag when its cycle completes.
%%
%% Accumulates fitness scores from all actuators before deciding
%% whether to continue or complete the evaluation.
loop(Id, ExoSelf_PId, SPIds, {[APId | APIds], MAPIds}, NPIds, CycleAcc, FitnessAcc, EFAcc, active) ->
    receive
        {APId, sync, Fitness, EndFlag} ->
            loop(Id, ExoSelf_PId, SPIds, {APIds, MAPIds}, NPIds, CycleAcc, FitnessAcc + Fitness, EFAcc + EndFlag, active);
        terminate ->
            % io:format("Cortex:~p is terminating.~n", [Id]),
            [PId ! {self(), terminate} || PId <- SPIds],
            [PId ! {self(), terminate} || PId <- MAPIds],
            [PId ! {self(), terminate} || PId <- NPIds]
    end;
%% @private
%% Active loop - all actuators synced
%%
%% Decides whether to:
%% - **EndFlag > 0**: Evaluation complete, report to ExoSelf and go inactive
%% - **EndFlag = 0**: Continue with next cycle
loop(Id, ExoSelf_PId, SPIds, {[], MAPIds}, NPIds, CycleAcc, FitnessAcc, EFAcc, active) ->
    case EFAcc > 0 of
        true ->
            TimeDif = timer:now_diff(os:timestamp(), get(start_time)),
            ExoSelf_PId ! {self(), evaluation_completed, FitnessAcc, CycleAcc, TimeDif},
            loop(Id, ExoSelf_PId, SPIds, {MAPIds, MAPIds}, NPIds, CycleAcc, FitnessAcc, EFAcc, inactive);
        false ->
            [PId ! {self(), sync} || PId <- SPIds],
            loop(Id, ExoSelf_PId, SPIds, {MAPIds, MAPIds}, NPIds, CycleAcc + 1, FitnessAcc, EFAcc, active)
    end;
%% @private
%% Inactive loop - waiting for reactivation
%%
%% After completing an evaluation, cortex enters inactive state.
%% Waits for ExoSelf to send reactivate message to start next evaluation.
%%
%% This allows ExoSelf to perturb neuron weights between evaluations.
loop(Id, ExoSelf_PId, SPIds, {MAPIds, MAPIds}, NPIds, _CycleAcc, _FitnessAcc, _EFAcc, inactive) ->
    receive
        {ExoSelf_PId, reactivate} ->
            put(start_time, os:timestamp()),
            [SPId ! {self(), sync} || SPId <- SPIds],
            loop(Id, ExoSelf_PId, SPIds, {MAPIds, MAPIds}, NPIds, 1, 0, 0, active);
        {ExoSelf_PId, terminate} ->
            ok
    end.
