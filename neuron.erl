-module(neuron).
-compile(export_all).
-include("records.hrl").

gen(ExoSelfPId, Node) ->
    spawn(Node, ?MODULE, loop, [ExoSelfPId]).

loop(ExoSelfPId) ->
    receive
        {ExoSelfPId, {Id, CortexPId, ActivationFunction, InputPIds, OutputPIds}} ->
            loop(Id, CortexPId, ActivationFunction, {InputPIds, InputPIds}, OutputPIds, 0)
    end.

loop(Id, CortexPId, ActivationFunction, {[{InputPId, Weights} | InputPIds], MInputPIds}, OutputPIds, Acc) ->
    receive
        {InputPId, forward, Input} ->
            Result = dot(Input, Weights, 0),
            loop(Id, CortexPId, ActivationFunction, {InputPIds, MInputPIds}, OutputPIds, Result + Acc);
        {CortexPId, backup} ->
            CortexPId ! {self(), Id, MInputPIds},
            loop(Id, CortexPId, ActivationFunction, {[{InputPId, Weights} | InputPIds], MInputPIds}, OutputPIds, Acc);
        {CortexPId, terminate} ->
            ok
    end;
loop(Id, CortexPId, ActivationFunction, {[Bias], MInputPIds}, OutputPIds, Acc) ->
    Output = neuron:activation_function(Acc + Bias),
    [OutputPId ! {self(), forward, [Output]} || OutputPId <- OutputPIds],
    loop(Id, CortexPId, ActivationFunction, {MInputPIds, MInputPIds}, OutputPIds, 0);
loop(Id, CortexPId, ActivationFunction, {[], MInputPIds}, OutputPIds, Acc) ->
    Output = neuron:activation_function(Acc),
    [OutputPId ! {self(), forward, [Output]} || OutputPId <- OutputPIds],
    loop(Id, CortexPId, ActivationFunction, {MInputPIds, MInputPIds}, OutputPIds, 0).


dot([I | Input], [W | Weights], Acc) ->
    dot(Input, Weights, I * W + Acc);
dot([], [], Acc) ->
    Acc.

tanh(Val) ->
    math:tanh(Val).
