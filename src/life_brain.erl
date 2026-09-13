%% One small, fixed feed-forward network, represented entirely as data.
%% Each row contains incoming weights followed by its bias.
-module(life_brain).
-export([new/0, forward/2, decide/2, action/1, mutate/1, weights/1]).

new() -> #{hidden => layer(6, 7), output => layer(4, 6)}.

layer(Count, Inputs) ->
    [[2*rand:uniform()-1 || _ <- lists:seq(1, Inputs+1)] || _ <- lists:seq(1, Count)].

forward(#{hidden := Hidden, output := Output}, Inputs) when length(Inputs) =:= 7 ->
    activate(Output, activate(Hidden, Inputs)).

activate(Layer, Inputs) -> [math:tanh(dot(Row, Inputs)) || Row <- Layer].
dot([Bias], []) -> Bias;
dot([Weight | Weights], [Value | Values]) -> Weight*Value + dot(Weights, Values).

decide(Brain, Inputs) -> action(forward(Brain, Inputs)).
action([North, East, South, West]) ->
    %% Des poids/scores différents peuvent donner exactement cette même action.
    %% Si le gagnant ne change jamais durant l'épisode, le trajet et le score
    %% restent identiques : voilà un plateau vu par notre recherche évolutive.
    %% Strict comparison gives a stable N/E/S/W tie break.
    {_, Action} = lists:foldl(fun({Score, Direction}, {Best, _}) when Score > Best ->
                                     {Score, Direction};
                                (_, Acc) -> Acc
                             end, {North, north}, [{East,east},{South,south},{West,west}]),
    Action.

weights(#{hidden := Hidden, output := Output}) -> lists:append(Hidden ++ Output).

mutate(#{hidden := Hidden, output := Output}) ->
    Forced = rand:uniform(76),
    {Rows, _} = lists:mapfoldl(fun(Row, Index) ->
        lists:mapfoldl(fun(Weight, I) ->
            New = case I =:= Forced orelse rand:uniform() < 0.15 of
                true -> max(-6.0, min(6.0, Weight + (2*rand:uniform()-1)*0.7));
                false -> Weight
            end,
            {New, I+1}
        end, Index, Row)
    end, 1, Hidden ++ Output),
    {NewHidden, NewOutput} = lists:split(6, Rows),
    #{hidden => NewHidden, output => NewOutput}.
