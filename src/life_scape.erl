%% Translate actor messages into the exact same pure food-world transitions.
-module(life_scape).
-export([run/2]).

run(Owner, Options) ->
    %% Une évaluation = un parcours de TOUS ces mondes avec les mêmes poids.
    %% Pour comparer deux méthodes, garder exactement cette liste et cet ordre.
    Seeds = maps:get(seeds, Options, [11,22,33]),
    true = is_list(Seeds) andalso Seeds =/= [],
    loop(Owner, Seeds, Seeds, life_world:new(hd(Seeds)), 0, [], Options).

loop(Owner, [Seed | Rest] = Remaining, Seeds, World, Score, Frames, Options) ->
    receive
        {Sensor, sense} ->
            Sensor ! {self(), percept, life_world:sense(World)},
            loop(Owner, Remaining, Seeds, World, Score, Frames, Options);
        {Actuator, action, Outputs} ->
            Action = life_brain:action(Outputs),
            Next = life_world:step(World, Action),
            Recording = maps:is_key(trace_to, Options),
            UpdatedFrames = case Recording of
                true -> [World#{inputs => life_world:sense(World), outputs => Outputs,
                                action => Action} | Frames];
                false -> []
            end,
            case maps:get(status, Next) of
                alive ->
                    Actuator ! {self(), 0, 0},
                    loop(Owner, Remaining, Seeds, Next, Score, UpdatedFrames, Options);
                Status ->
                    #{eaten := Eaten, tick := Steps} = Next,
                    EpisodeScore = 100*Eaten + Steps,
                    case Recording of
                        true ->
                            Final = Next#{inputs => life_world:sense(Next), outputs => [], action => none},
                            Episode = #{score => EpisodeScore, eaten => Eaten, steps => Steps,
                                        status => Status, frames => lists:reverse([Final | UpdatedFrames])},
                            maps:get(trace_to, Options) ! {life_trace, maps:get(trace_ref, Options), Seed, Episode};
                        false -> ok
                    end,
                    case Rest of
                        [] ->
                            %% Fitness moyen, Halt=1 : le cortex peut terminer
                            %% l'évaluation et ExoSelf peut ajuster les poids.
                            %% Les Halt=0 précédents poursuivaient le même essai.
                            Actuator ! {self(), (Score+EpisodeScore)/length(Seeds), 1},
                            loop(Owner, Seeds, Seeds, life_world:new(hd(Seeds)), 0, [], Options);
                        [NextSeed | _] ->
                            Actuator ! {self(), 0, 0},
                            loop(Owner, Rest, Seeds, life_world:new(NextSeed), Score+EpisodeScore, [], Options)
                    end
            end;
        {Owner, terminate} -> ok
    end.
