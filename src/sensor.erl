%%% # Sensor Module
%%%
%%% Sensors are the **input layer** of the neural network. Each sensor
%%% runs as an independent process that retrieves data from the environment
%%% (scapes) and forwards it to connected neurons.
%%%
%%% ## Responsibilities
%%%
%%% - **Data Acquisition**: Request sensory input from scapes
%%% - **Broadcasting**: Forward input vectors to all connected neurons (fanout)
%%% - **Synchronization**: Triggered by cortex to start each evaluation cycle
%%% - **Error Handling**: Validate input vectors and handle scape timeouts
%%%
%%% ## Process Lifecycle
%%%
%%% 1. **Spawning**: Created by ExoSelf via `gen/2'
%%% 2. **Initialization**: Receives connection info (ID, cortex PID, scape, sensor function, fanout neurons)
%%% 3. **Processing Loop**: Waits for sync messages, gets percepts, broadcasts to neurons
%%% 4. **Termination**: Receives terminate message and exits
%%%
%%% ## Message Protocol
%%%
%%% **Input Messages**:
%%% - `{CortexPId, sync}' - Signal to sense and forward
%%% - `{CortexPId, terminate}' - Shutdown sensor process
%%%
%%% **Output Messages**:
%%% - `{ScapePId, sense}' - Request sensory input from scape
%%% - `{self(), forward, SensoryVector}' - Send input to neurons
%%%
%%% **Scape Response**:
%%% - `{ScapePId, percept, SensoryVector}' - Sensory data from environment
%%%
%%% ## Sensor Types
%%%
%%% Different sensor functions can be plugged in via the morphology:
%%% - `xor_GetInput/2' - Gets XOR problem inputs from scape
%%% - `rng/1' - Generates random input (for testing)
%%% - Custom sensor functions can be added as needed

-module(sensor).
-compile(export_all).
-include("records.hrl").

-export([gen/2, rng/1, xor_GetInput/2]).

%%==============================================================================
%% API Functions
%%==============================================================================

%% @doc Spawn a new sensor process
%%
%% Creates a sensor process on the specified node. The sensor enters
%% an initialization state waiting for configuration from ExoSelf.
%%
%% === Parameters ===
%% - `ExoSelfPId' - PID of the ExoSelf orchestrator
%% - `Node' - Node where the sensor should be spawned
%%
%% === Returns ===
%% PID of the spawned sensor process
%%
%% === Examples ===
%% ```
%% SensorPId = sensor:gen(ExoSelfPId, node()).
%% '''
-spec gen(pid(), node()) -> pid().
gen(ExoSelfPId, Node) ->
    spawn(Node, ?MODULE, loop, [ExoSelfPId]).

%%==============================================================================
%% Sensor Functions
%%==============================================================================

%% @doc Random number generator sensor
%%
%% Generates a vector of random floating-point numbers in range [0, 1).
%% Useful for testing network behavior without a scape.
%%
%% === Parameters ===
%% - `VL' - Vector Length (number of random values to generate)
%%
%% === Returns ===
%% List of VL random floats
%%
%% === Examples ===
%% ```
%% sensor:rng(3).
%% [0.42, 0.87, 0.15]
%% '''
-spec rng(VL :: pos_integer()) -> [float()].
rng(VL) ->
    rng(VL, []).

%% @private
rng(0, Acc) ->
    Acc;
rng(VL, Acc) ->
    rng(VL - 1, [rand:uniform() | Acc]).

%% @doc XOR problem input sensor
%%
%% Retrieves input vectors from an XOR scape. Sends a `sense' message
%% to the scape and waits for a `percept' response.
%%
%% === Error Handling ===
%% - **Vector length mismatch**: Returns zero vector if scape returns wrong size
%% - **Scape timeout**: Returns zero vector after 5 seconds with warning
%%
%% === Parameters ===
%% - `VL' - Expected vector length (should be 2 for XOR)
%% - `Scape' - PID of the scape process
%%
%% === Returns ===
%% Sensory vector from scape, or zero vector on error
%%
%% === Protocol ===
%% ```
%% Sensor -> Scape: {self(), sense}
%% Scape -> Sensor: {ScapePId, percept, [Input1, Input2]}
%% '''
%%
%% === Examples ===
%% ```
%% sensor:xor_GetInput(2, ScapePId).
%% [-1.0, 1.0]
%% '''
-spec xor_GetInput(VL :: pos_integer(), Scape :: pid()) -> [float()].
xor_GetInput(VL, Scape) ->
    Scape ! {self(), sense},
    receive
        {Scape, percept, SensoryVector} ->
            case length(SensoryVector) == VL of
                true ->
                    SensoryVector;
                false ->
                    io:format("Error in sensor:xor_GetInput/2, VL:~p SensoryVector: ~p~n",
                             [VL, SensoryVector]),
                    lists:duplicate(VL, 0)
            end
    after 5000 ->
        io:format("Sensor: Timeout waiting for scape~n"),
        lists:duplicate(VL, 0)
    end.

%%==============================================================================
%% Internal Functions - Processing Loop
%%==============================================================================

%% @private
%% Initialization loop - wait for configuration
loop(ExoSelfPId) ->
    receive
        {ExoSelfPId, {Id, CortexPId, Scape, SensorName, VL, FanoutPIds}} ->
            loop(Id, CortexPId, Scape, SensorName, VL, FanoutPIds)
    end.

%% @private
%% Main processing loop
%%
%% Waits for sync signals from cortex, retrieves sensory input,
%% and broadcasts to all connected neurons.
%%
%% The sensor function is called dynamically: `sensor:SensorName(VL, Scape)'
loop(Id, CortexPId, Scape, SensorName, VL, FanoutPIds) ->
    receive
        {CortexPId, sync} ->
            SensoryVector = sensor:SensorName(VL, Scape),
            [Pid ! {self(), forward, SensoryVector} || Pid <- FanoutPIds],
            loop(Id, CortexPId, Scape, SensorName, VL, FanoutPIds);
        {CortexPId, terminate} ->
            ok
    end.
