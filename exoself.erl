-module(exoself).
-compile(export_all).

-include("records.hrl").

map() ->
  map(ffnn).

map(FileName) ->
  {ok, Genotype} = file:consult(FileName),
  spawn(?MODULE, map, [FileName, Genotype]).

map(FileName, Genotype) ->
  IdsNPIds = ets:new(idsNpids, [set, private]),
  [Cx | CerebralUnits] = Genotype,
  SensorIds = Cx#cortex.sensor_ids,
  ActuatorIds = Cx#cortex.actuator_ids,
  NIds = Cx#cortex.neuron_ids,
  spawn_CerebralUnits(IdsNPIds, cortex, [Cx#cortex.id]),
  spawn_CerebralUnits(IdsNPIds, sensor, SensorIds),
  spawn_CerebralUnits(IdsNPIds, actuator, ActuatorIds),
  spawn_CerebralUnits(IdsNPIds, neuron, NIds),
  link_CerebralUnits(CerebralUnits, IdsNPIds),
  link_Cortex(Cx, IdsNPIds),
  Cx_PId = ets:lookup_element(IdsNPIds, Cx#cortex.id, 2),
  receive
    {Cx_PId, backup, Neuron_IdsNWeights} ->
      U_Genotype = update_genotype(IdsNPIds, Genotype, Neuron_IdsNWeights),
      {ok, File} = file:open(FileName, write),
      lists:foreach(fun(X) -> io:format(File, "~p~n", [X]) end, U_Genotype),
      file:close(File),
      io:format("Finished updating to file: ~p~n", [FileName])
  end.

spawn_CerebralUnits(IdsNPIds, CereralUnitType, [Id | Ids]) ->
  PId = CereralUnitType:gen(self(), node()),
  ets:insert(IdsNPIds, {Id, PId}),
  ets:insert(IdsNPIds, {PId, Id}),
  spawn_CerebralUnits(IdsNPIds, CereralUnitType, Ids);
spawn_CerebralUnits(_IdsNPIds, _CereralUnitType, []) ->
  true.

link_CerebralUnits([R | Records], IdsNPIds) when is_record(R, sensor) ->
  SId = R#sensor.id,
  SPId = ets:lookup_element(IdsNPIds, SId, 2),
  Cx_PId = ets:lookup_element(IdsNPIds, R#sensor.cortex_id, 2),
  SName = R#sensor.name,
  Fanout_Ids = R#sensor.fanout_ids,
  Fanout_PIds = [ets:lookup_element(IdsNPIds, Id, 2) || Id <- Fanout_Ids],
  SPId ! {self(), {SId, Cx_PId, SName, R#sensor.vector_length, Fanout_PIds}},
  link_CerebralUnits(Records, IdsNPIds);
link_CerebralUnits([R | Records], IdsNPIds) when is_record(R, actuator) ->
  AId = R#actuator.id,
  APId = ets:lookup_element(IdsNPIds, AId, 2),
  Cx_PId = ets:lookup_element(IdsNPIds, R#actuator.cortex_id, 2),
  AName = R#actuator.name,
  Fanin_Ids = R#actuator.fanin_ids,
  Fanin_PIds = [ets:lookup_element(IdsNPIds, Id, 2) || Id <- Fanin_Ids],
  APId ! {self(), {AId, Cx_PId, AName, Fanin_PIds}},
  link_CerebralUnits(Records, IdsNPIds);
link_CerebralUnits([R | Records], IdsNPIds) when is_record(R, neuron) ->
  NId = R#neuron.id,
  NPId = ets:lookup_element(IdsNPIds, NId, 2),
  Cx_PId = ets:lookup_element(IdsNPIds, R#neuron.cortex_id, 2),
  AFName = R#neuron.activation_function,
  Input_IdPs = R#neuron.input_ids,
  Output_Ids = R#neuron.output_ids,
  Input_PIdPs = convert_neuron_weights_to_process_weights(IdsNPIds, Input_IdPs, []),
  Output_PIds = [ets:lookup_element(IdsNPIds, Id, 2) || Id <- Output_Ids],
  NPId ! {self(), {NId, Cx_PId, AFName, Input_PIdPs, Output_PIds}},
  link_CerebralUnits(Records, IdsNPIds);
link_CerebralUnits([], _IdsNPIds) ->
  ok.

convert_neuron_weights_to_process_weights(_IdsNPIds,[{bias,Bias}],Acc)->
    lists:reverse([Bias|Acc]);
convert_neuron_weights_to_process_weights(IdsNPIds,[{Id,Weights}|Fanin_IdPs],Acc)->
    convert_neuron_weights_to_process_weights(IdsNPIds,Fanin_IdPs,
        [{ets:lookup_element(IdsNPIds,Id,2),Weights}|Acc]).

link_Cortex(Cx, IdsNPIds) ->
  Cx_Id = Cx#cortex.id,
  Cx_PId = ets:lookup_element(IdsNPIds, Cx_Id, 2),
  SIds = Cx#cortex.sensor_ids,
  AIds = Cx#cortex.actuator_ids,
  NIds = Cx#cortex.neuron_ids,
  SPIds = [ets:lookup_element(IdsNPIds, SId, 2) || SId <- SIds],
  APIds = [ets:lookup_element(IdsNPIds, AId, 2) || AId <- AIds],
  NPIds = [ets:lookup_element(IdsNPIds, NId, 2) || NId <- NIds],
  Cx_PId ! {self(), {Cx_Id, SPIds, APIds, NPIds}, 1000}.

update_genotype(IdsNPIds, Genotype, [{N_Id, PIdPs} | WeightPs]) ->
  N = lists:keyfind(N_Id, 2, Genotype),
  io:format("Process Ids: ~p~n", [PIdPs]),
  Updated_InputIdPs = convert_process_weights_to_neuron_weights(IdsNPIds, PIdPs, []),
  Updated_Neuron = N#neuron{input_ids = Updated_InputIdPs},
  Updated_Genotype = lists:keyreplace(N_Id, 2, Genotype, Updated_Neuron),
  io:format("Neuron: ~p~nUpdated Neuron: ~p~nGenotype: ~p~nUpdated Genotype: ~p~n", [N, Updated_Neuron, Genotype, Updated_Genotype]),
  update_genotype(IdsNPIds, Updated_Genotype, WeightPs);
update_genotype(_IdsNPIds, Genotype, []) ->
  Genotype.

convert_process_weights_to_neuron_weights(IdsNPIds,[{PId,Weights}|Input_PIdPs],Acc) ->
    convert_process_weights_to_neuron_weights(
        IdsNPIds,
        Input_PIdPs,
        [{ets:lookup_element(IdsNPIds,PId,2),Weights}|Acc]
    );
convert_process_weights_to_neuron_weights(_IdsNPIds,[Bias],Acc) ->
    lists:reverse([{bias,Bias}|Acc]).
