-module(sensor).
-compile(export_all).
-include("records.hrl").

gen(ExoSelfPId, Node) ->
    spawn(Node, ?MODULE, loop, [ExoSelfPId]).

loop(ExoSelfPId) ->
    receive
        {ExoSelfPId, {Id, CortexPId, SensorName, VL, FanoutPIds}} ->
            loop(Id, CortexPId, SensorName, VL, FanoutPIds)
    end.

loop(Id, CortexPId, SensorName, VL, FanoutPIds) ->
    receive
        {CortexPId, sync} ->
            SensoryVector = sensor:SensorName(VL),
            [Pid ! {self(), forward, SensoryVector} || Pid <- FanoutPIds],
            loop(Id, CortexPId, SensorName, VL, FanoutPIds);
        {CortexPId, terminate} ->
            ok
        end.

rng(VL) ->
    rng(VL, []).
rng(0, Acc) ->
    Acc;
rng(VL, Acc) ->
    rng(VL - 1, [rand:uniform() | Acc]).

xor_getInput(VL, Scape) ->
    Scape ! {self(), sense},
    receive
        {Scape, percept, SensoryVector} ->
            case length(SensoryVector) == VL of
                true -> SensoryVector;
                false ->
                    io:format("Error in sensor:xor_sim/2, VL:~p SensoryVector: ~p~n", [VL, SensoryVector]),
                    lists:duplicate(VL, 0)
            end
    end.