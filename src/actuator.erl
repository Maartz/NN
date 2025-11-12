%%% # Actuator Module
%%%
%%% Actuators are the **output layer** of the neural network. Each actuator
%%% runs as an independent process that collects outputs from neurons,
%%% sends actions to scapes, and reports fitness back to the cortex.
%%%
%%% ## Responsibilities
%%%
%%% - **Output Collection**: Gather outputs from all connected neurons (fanin)
%%% - **Action Execution**: Send collected outputs to scape for evaluation
%%% - **Fitness Reporting**: Receive fitness score and halt flag from scape
%%% - **Synchronization**: Signal cortex when evaluation cycle completes
%%%
%%% ## Process Lifecycle
%%%
%%% 1. **Spawning**: Created by ExoSelf via `gen/2'
%%% 2. **Initialization**: Receives connection info (ID, cortex PID, scape, actuator function, fanin neurons)
%%% 3. **Processing Loop**: Collects neuron outputs, sends to scape, reports fitness
%%% 4. **Termination**: Receives terminate message and exits
%%%
%%% ## Message Protocol
%%%
%%% **Input Messages**:
%%% - `{NeuronPId, forward, Output}' - Output value from connected neuron
%%% - `{ExoSelf_PId, terminate}' - Shutdown actuator process
%%%
%%% **Output Messages**:
%%% - `{self(), action, OutputVector}' - Send action to scape
%%% - `{self(), sync, Fitness, EndFlag}' - Report results to cortex
%%%
%%% **Scape Response**:
%%% - `{ScapePId, Fitness, HaltFlag}' - Fitness score and continue/halt signal
%%%
%%% ## Actuator Types
%%%
%%% Different actuator functions can be plugged in via the morphology:
%%% - `xor_SendOutput/2' - Sends XOR output to scape and gets fitness
%%% - `pts/2' - Prints output to screen (for debugging/testing)
%%% - Custom actuator functions can be added as needed
%%%
%%% ## Fitness and Halt Flags
%%%
%%% **Fitness**: Higher values = better performance (problem-specific)
%%% **HaltFlag**:
%%% - `1' - Evaluation complete, trigger next learning cycle
%%% - `0' - Continue evaluation with more inputs

-module(actuator).
-compile(export_all).
-include("records.hrl").

-export([gen/2, pts/2, xor_SendOutput/2]).

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Spawn a new actuator process
%%
%% Creates an actuator process on the specified node. The actuator enters
%% a preparation state waiting for initialization data from ExoSelf.
%%
%% === Parameters ===
%% - `ExoSelf_PId' - PID of the ExoSelf orchestrator
%% - `Node' - Node where the actuator should be spawned
%%
%% === Returns ===
%% PID of the spawned actuator process
%%
%% === Examples ===
%% ```
%% ActuatorPId = actuator:gen(ExoSelfPId, node()).
%% '''
-spec gen(pid(), node()) -> pid().
gen(ExoSelf_PId, Node) ->
    spawn(Node, ?MODULE, prep, [ExoSelf_PId]).

%%==============================================================================
%% Actuator Functions
%%==============================================================================

%% @doc Print-to-screen actuator
%%
%% Debugging actuator that prints the output vector to the console
%% and returns a dummy fitness of 1 with halt flag 0 (continue).
%%
%% Useful for observing network outputs during development.
%%
%% === Parameters ===
%% - `Result' - Output vector from neurons
%% - `_Scape' - Scape PID (unused)
%%
%% === Returns ===
%% Tuple `{Fitness, HaltFlag}' - Always {1, 0}
%%
%% === Examples ===
%% ```
%% actuator:pts([0.5, -0.3], ScapePId).
%% actuator:pts(Result): [0.5, -0.3]
%% {1, 0}
%% '''
-spec pts(Result :: [float()], Scape :: pid()) -> {Fitness :: number(), HaltFlag :: 0 | 1}.
pts(Result, _Scape) ->
    io:format("actuator:pts(Result): ~p~n", [Result]),
    {1, 0}.

%% @doc XOR problem output actuator
%%
%% Sends the network's output to the XOR scape for evaluation.
%% Receives a fitness score and halt flag in response.
%%
%% === Protocol ===
%% ```
%% Actuator -> Scape: {self(), action, Output}
%% Scape -> Actuator: {ScapePId, Fitness, HaltFlag}
%% '''
%%
%% === Parameters ===
%% - `Output' - Output vector from neurons (should be length 1 for XOR)
%% - `Scape' - PID of the XOR scape process
%%
%% === Returns ===
%% Tuple `{Fitness, HaltFlag}' where:
%% - **Fitness**: 1/(MSE + 0.00001), higher is better
%% - **HaltFlag**: 1 when all 4 XOR cases evaluated, 0 otherwise
%%
%% === Examples ===
%% ```
%% actuator:xor_SendOutput([0.8], ScapePId).
%% {145.67, 1}
%% '''
-spec xor_SendOutput(Output :: [float()], Scape :: pid()) -> {Fitness :: float(), HaltFlag :: 0 | 1}.
xor_SendOutput(Output, Scape) ->
    Scape ! {self(), action, Output},
    receive
        {Scape, Fitness, HaltFlag} ->
            {Fitness, HaltFlag}
    end.

%%==============================================================================
%% Internal Functions - Processing Loop
%%==============================================================================

%% @private
%% Preparation loop - wait for initialization
prep(ExoSelf_PId) ->
    receive
        {ExoSelf_PId, {Id, Cx_PId, Scape, ActuatorName, Fanin_PIds}} ->
            loop(Id, ExoSelf_PId, Cx_PId, Scape, ActuatorName, {Fanin_PIds, Fanin_PIds}, [])
    end.

%% @private
%% Main processing loop - collecting neuron outputs
%%
%% Accumulates outputs from all fanin neurons. Once all are received,
%% sends the complete output vector to the scape via the actuator function.
%%
%% The actuator function is called dynamically: `actuator:AName(Output, Scape)'
loop(Id, ExoSelf_PId, Cx_PId, Scape, AName, {[From_PId | Fanin_PIds], MFanin_PIds}, Acc) ->
    receive
        {From_PId, forward, Input} ->
            loop(Id, ExoSelf_PId, Cx_PId, Scape, AName, {Fanin_PIds, MFanin_PIds}, lists:append(Input, Acc));
        {ExoSelf_PId, terminate} ->
            ok
    end;
%% @private
%% Processing loop - all inputs received
%%
%% Sends output to scape, receives fitness and halt flag,
%% reports to cortex, and resets for next cycle.
loop(Id, ExoSelf_PId, Cx_PId, Scape, AName, {[], MFanin_PIds}, Acc) ->
    {Fitness, EndFlag} = actuator:AName(lists:reverse(Acc), Scape),
    % io:format("Actuator ~p: Got fitness ~p, endflag ~p~n", [Id, Fitness, EndFlag]),
    Cx_PId ! {self(), sync, Fitness, EndFlag},
    loop(Id, ExoSelf_PId, Cx_PId, Scape, AName, {MFanin_PIds, MFanin_PIds}, []).
