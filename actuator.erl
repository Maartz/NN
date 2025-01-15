-module(actuator).
-compile(export_all).
-include("records.hrl").

gen(ExoSelfPId, Node) ->
    spawn(Node, ?MODULE, loop, [ExoSelfPId]).

loop(ExoSelfPId) ->
    receive
        {ExoSelfPId, {Id, CortexPId, ActuatorName, FaninPIds}} ->
            loop(Id, CortexPId, ActuatorName, {FaninPIds, FaninPIds}, [])
    end.

loop(Id, CortexPId, ActuatorName, {[FromPId | FaninPIds], MFaninPIds}, Acc) ->
    receive
        {FromPId, forward, Input} ->
            loop(Id, CortexPId, ActuatorName, {FaninPIds, MFaninPIds}, lists:append(Input, Acc));
        {CortexPId, terminate} ->
            ok
        end;
loop(Id, CortexPId, ActuatorName, {[], MFaninPIds}, Acc) ->
    actuator:ActuatorName(lists:reverse(Acc)),
    CortexPId ! {self(), sync},
    loop(Id, CortexPId, ActuatorName, {MFaninPIds, MFaninPIds}, []).

pts(Result) ->
    io:format("actuators:pts(Result): ~p~n", [Result]).
