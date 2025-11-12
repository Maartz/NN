%%% # Scape Module
%%%
%%% Scapes are **environment simulators** that provide training/testing
%%% environments for neural networks. Each scape runs as an independent
%%% process that supplies input data, evaluates outputs, and calculates fitness.
%%%
%%% ## Responsibilities
%%%
%%% - **Environment Simulation**: Model the problem domain (e.g., XOR logic)
%%% - **Input Generation**: Provide training data to sensors via percepts
%%% - **Output Evaluation**: Compare network outputs against expected results
%%% - **Fitness Calculation**: Compute performance metrics (MSE, accuracy, etc.)
%%% - **Cycle Management**: Track evaluation progress and signal completion
%%%
%%% ## Process Lifecycle
%%%
%%% 1. **Spawning**: Created by ExoSelf via `gen/2'
%%% 2. **Initialization**: Receives scape name and calls specific scape function
%%% 3. **Simulation Loop**: Handles sense/action message pairs
%%% 4. **Termination**: Receives terminate message and exits
%%%
%%% ## Message Protocol
%%%
%%% **Input Messages**:
%%% - `{SensorPId, sense}' - Sensor requests input data
%%% - `{ActuatorPId, action, Output}' - Actuator sends network output for evaluation
%%% - `{ExoSelf_PId, terminate}' - Shutdown scape process
%%%
%%% **Output Messages**:
%%% - `{self(), percept, InputVector}' - Send input data to sensor
%%% - `{self(), Fitness, HaltFlag}' - Report fitness score to actuator
%%%
%%% ## Scape Types
%%%
%%% ### Private vs Public Scapes
%%% - **Private**: Each agent spawns its own scape instance (e.g., xor_sim)
%%% - **Public**: Multiple agents share a single scape (for competitive/cooperative scenarios)
%%%
%%% ### XOR Scape (xor_sim)
%%% Provides the classic XOR problem for testing non-linearly separable learning:
%%% - **Training set**: 4 cases ([-1,-1]→[-1], [1,-1]→[1], [-1,1]→[1], [1,1]→[-1])
%%% - **Fitness**: 1 / (MSE + 0.00001), where MSE is mean squared error
%%% - **Halt**: Signals completion after all 4 cases evaluated
%%%
%%% ## Evaluation Cycle
%%%
%%% 1. Sensor sends `{sense}' → Scape responds with input vector
%%% 2. Network processes input through layers
%%% 3. Actuator sends `{action, Output}' → Scape evaluates output
%%% 4. Scape calculates error and sends fitness + halt flag
%%% 5. If more cases remain: continue with next input (goto step 1)
%%% 6. If all cases complete: send final fitness with halt=1

-module(scape).
-compile(export_all).
-include("records.hrl").

-export([gen/2, xor_sim/1]).

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Spawn a new scape process
%%
%% Creates a scape process on the specified node. The scape enters
%% a preparation state waiting for its name/type from ExoSelf.
%%
%% === Parameters ===
%% - `ExoSelf_PId' - PID of the ExoSelf orchestrator
%% - `Node' - Node where the scape should be spawned
%%
%% === Returns ===
%% PID of the spawned scape process
%%
%% === Examples ===
%% ```
%% ScapePId = scape:gen(ExoSelfPId, node()).
%% '''
-spec gen(pid(), node()) -> pid().
gen(ExoSelf_PId, Node) ->
    spawn(Node, ?MODULE, prep, [ExoSelf_PId]).

%%==============================================================================
%% Internal Functions - Processing Loop
%%==============================================================================

%% @private
%% Preparation loop - wait for scape name
%%
%% Receives the name of the scape function to run and calls it.
%% This allows dynamic selection of environment types.
prep(ExoSelf_PId) ->
    receive
        {ExoSelf_PId, Name} ->
            scape:Name(ExoSelf_PId)
    end.

%%==============================================================================
%% Scape Implementations
%%==============================================================================

%% @doc XOR problem simulator
%%
%% Provides the classic XOR (exclusive OR) problem environment.
%% XOR is a canonical non-linearly separable problem used to test
%% whether a network can learn beyond simple linear classification.
%%
%% === Truth Table ===
%% ```
%% Input         | Output
%% --------------|-------
%% [-1, -1]      | [-1]
%% [ 1, -1]      | [ 1]
%% [-1,  1]      | [ 1]
%% [ 1,  1]      | [-1]
%% '''
%%
%% === Fitness Calculation ===
%% After all 4 cases are evaluated:
%% - **MSE** = sqrt(sum of squared errors)
%% - **Fitness** = 1 / (MSE + 0.00001)
%% - Higher fitness = better performance
%% - Perfect solution achieves fitness ≈ 100000
%%
%% === Evaluation Flow ===
%% 1. Receives `{sense}' from sensor → sends first input pair
%% 2. Receives `{action, Output}' from actuator → calculates error
%% 3. Repeats for all 4 XOR cases
%% 4. After 4th case: sends final fitness with HaltFlag=1
%% 5. Resets for next evaluation cycle
%%
%% === Parameters ===
%% - `ExoSelf_PId' - PID of the ExoSelf process (for termination)
%%
%% === Examples ===
%% ```
%% % Spawned by ExoSelf
%% ScapePId = scape:gen(ExoSelfPId, node()),
%% ScapePId ! {ExoSelfPId, xor_sim}.
%% '''
-spec xor_sim(pid()) -> ok.
xor_sim(ExoSelf_PId) ->
    XOR = [{[-1, -1], [-1]}, {[1, -1], [1]}, {[-1, 1], [1]}, {[1, 1], [-1]}],
    xor_sim(ExoSelf_PId, {XOR, XOR}, 0).

%% @private
%% XOR simulator main loop
%%
%% Maintains state:
%% - **Remaining cases**: Cases left in current evaluation cycle
%% - **Master XOR**: Full set of 4 cases (for reset)
%% - **Error accumulator**: Sum of squared errors
%%
%% === Message Handling ===
%% - `{sense}': Sensor requesting input → send current input vector
%% - `{action, Output}': Actuator sending result → evaluate and update
%% - `{terminate}': Shutdown signal from ExoSelf
%%
%% === Halt Conditions ===
%% - **HaltFlag=0**: More cases remaining, continue evaluation
%% - **HaltFlag=1**: All cases evaluated, send final fitness
-spec xor_sim(pid(), {{list(), list()}, list()}, float()) -> ok.
xor_sim(ExoSelf_PId, {[{Input, CorrectOutput} | XOR], MXOR}, ErrAcc) ->
    receive
        {From, sense} ->
            From ! {self(), percept, Input},
            xor_sim(ExoSelf_PId, {[{Input, CorrectOutput} | XOR], MXOR}, ErrAcc);
        {From, action, Output} ->
            Error = list_compare(Output, CorrectOutput, 0),
            case XOR of
                [] ->
                    % All 4 cases evaluated - calculate final fitness
                    MSE = math:sqrt(ErrAcc + Error),
                    Fitness = 1 / (MSE + 0.00001),
                    From ! {self(), Fitness, 1},
                    xor_sim(ExoSelf_PId, {MXOR, MXOR}, 0);
                _ ->
                    % More cases remaining - continue without fitness
                    From ! {self(), 0, 0},
                    xor_sim(ExoSelf_PId, {XOR, MXOR}, ErrAcc + Error)
            end;
        {ExoSelf_PId, terminate} ->
            ok
    end.

%% @private
%% Calculate euclidean distance between two vectors
%%
%% Computes the root mean squared error (RMSE) between the network's
%% output and the expected target output.
%%
%% === Formula ===
%% ```
%% distance = sqrt(sum((x_i - y_i)^2))
%% '''
%%
%% === Parameters ===
%% - `List1' - Network output vector
%% - `List2' - Expected output vector
%% - `ErrorAcc' - Accumulator for squared error sum
%%
%% === Returns ===
%% Float representing the euclidean distance (RMSE)
-spec list_compare(list(float()), list(float()), float()) -> float().
list_compare([X | List1], [Y | List2], ErrorAcc) ->
    list_compare(List1, List2, ErrorAcc + math:pow(X - Y, 2));
list_compare([], [], ErrorAcc) ->
    math:sqrt(ErrorAcc).
