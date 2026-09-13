%% A complete world is an ordinary value. No mailbox, database or hidden RNG.
-module(life_world).
-export([new/1, sense/1, step/2, episode/2, evaluate/2, scripted_action/1]).

new(Seed) ->
    Size = 12,
    Position = {Size div 2, Size div 2},
    Cells = [{X,Y} || X <- lists:seq(0, Size-1), Y <- lists:seq(0, Size-1),
                      {X,Y} =/= Position],
    {Ranked, _} = lists:mapfoldl(fun(Cell, Random) ->
        {Key, Next} = rand:uniform_s(Random),
        {{Key, Cell}, Next}
    end, rand:seed_s(exsplus, Seed), Cells),
    Food = [Cell || {_, Cell} <- lists:sublist(lists:sort(Ranked), 14)],
    #{size => Size, position => Position, food => Food, energy => 40,
      capacity => 60, tick => 0, eaten => 0, status => alive, event => start}.

%% Direction to the nearest food, energy, then walls N/E/S/W.
%% This is a deliberately generous sensor, not simulated vision.
sense(#{position := {X,Y}, size := Size, energy := Energy, capacity := Capacity} = W) ->
    {FX,FY} = nearest_food(W),
    [(FX-X)/(Size-1), (FY-Y)/(Size-1), Energy/Capacity,
     flag(Y =:= 0), flag(X =:= Size-1), flag(Y =:= Size-1), flag(X =:= 0)].

flag(true) -> 1.0;
flag(false) -> 0.0.

nearest_food(#{food := [], position := Position}) -> Position;
nearest_food(#{food := Food, position := {X,Y}}) ->
    {_, Position} = lists:min([{abs(FX-X)+abs(FY-Y), {FX,FY}} || {FX,FY} <- Food]),
    Position.

scripted_action(#{position := {X,Y}} = World) ->
    {FX,FY} = nearest_food(World),
    if FX > X -> east; FX < X -> west; FY > Y -> south; true -> north end.

step(#{status := Status} = W, _) when Status =/= alive -> W;
step(#{position := {X,Y}, size := Size, food := Food, energy := Energy,
       capacity := Capacity, eaten := Eaten, tick := Tick} = W, Action) ->
    {DX,DY} = direction(Action),
    Candidate = {X+DX,Y+DY},
    Blocked = X+DX < 0 orelse X+DX >= Size orelse Y+DY < 0 orelse Y+DY >= Size,
    Position = case Blocked of true -> {X,Y}; false -> Candidate end,
    Ate = lists:member(Position, Food),
    {NextFood, NextEnergy, NextEaten, Event} = case Ate of
        true -> {lists:delete(Position, Food), min(Capacity, Energy-1+18), Eaten+1, ate};
        false -> {Food, Energy-1, Eaten, case Blocked of true -> wall; false -> moved end}
    end,
    Status = if NextFood =:= [] -> cleared;
                NextEnergy =< 0 -> exhausted;
                Tick+1 >= 160 -> limit;
                true -> alive end,
    W#{position => Position, food => NextFood, energy => NextEnergy,
       eaten => NextEaten, tick => Tick+1, event => Event, status => Status}.

direction(north) -> {0,-1};
direction(east) -> {1,0};
direction(south) -> {0,1};
direction(west) -> {-1,0}.

%% Only replays retain frames; evolution uses evaluate/2 to avoid this allocation.
episode(Controller, Seed) -> run(Controller, new(Seed), [], true).
evaluate(Controller, Seed) -> run(Controller, new(Seed), [], false).

run(Controller, #{status := alive} = World, Frames, Record) ->
    Inputs = sense(World),
    {Action, Outputs} = case Controller of
        scripted -> {scripted_action(World), []};
        _ ->
            Values = life_brain:forward(Controller, Inputs),
            {life_brain:action(Values), Values}
    end,
    NextFrames = case Record of
        true -> [World#{inputs => Inputs, outputs => Outputs, action => Action} | Frames];
        false -> Frames
    end,
    run(Controller, step(World, Action), NextFrames, Record);
run(_, World, Frames, Record) ->
    #{eaten := Eaten, tick := Steps, status := Status} = World,
    Result = #{score => 100*Eaten + Steps, eaten => Eaten, steps => Steps, status => Status},
    case Record of
        true -> Result#{frames => lists:reverse([
                    World#{inputs => sense(World), outputs => [], action => none} | Frames])};
        false -> Result
    end.
