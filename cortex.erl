-module(cortex).
-compile(export_all).
-include("records.hrl").

gen(ExoSelfPId, Node) ->
    spawn(Node, ?MODULE, loop, [ExoSelfPId]).


loop(ExoSelfPId) ->
    receive
        {ExoSelfPId, {Id, SPIds, APIds, NPIds}, TotalSteps} ->
            [SPId ! {self(), sync} || SPId <- SPIds],
            loop(Id, ExoSelfPId, SPIds, {APIds, APIds}, NPIds, TotalSteps)
    end.

loop(Id, ExoSelfPId, SPIds, {_APIds, MAPIds}, NPIds, 0) ->
    io:format("Cortex ~p is backing up and terminating.~n", [Id]),
    NeuronIDsWeights = get_backup(NPIds, []),
    ExoSelfPId ! {self(), bakcup, NeuronIDsWeights},
    [PId ! {self(), terminate} || PId <- SPIds],
    [PId ! {self(), terminate} || PId <- MAPIds],
    [PId ! {self(), terminate} || PId <- NPIds];
loop(Id, ExoSelfPId, SPIds, {[APId | APIds], MAPIds}, NPIds, Step) ->
    receive
        {APId, sync} ->
            loop(Id, ExoSelfPId, SPIds, {APIds, MAPIds}, NPIds, Step);
        terminate ->
            io:format("Cortex: ~p is terminating~n", [Id]),
            [PId ! {self(), terminate} || PId <- SPIds],
            [PId ! {self(), terminate} || PId <- MAPIds],
            [PId ! {self(), terminate} || PId <- NPIds]
    end;
loop(Id, ExoSelfPId, SPIds, {[], MAPIds}, NPIds, Step) ->
    [PId ! {self(), sync} || PId <- SPIds],
    loop(Id, ExoSelfPId, SPIds, {MAPIds, MAPIds}, NPIds, Step - 1).

get_backup([NPId | NPIds], Acc) ->
    NPId ! {self(), get_backup},
    receive
        {NPId, NId, WeightTuples} ->
            get_backup(NPIds, [{NId, WeightTuples} | Acc])
    end;
get_backup([], Acc) ->
    Acc.
