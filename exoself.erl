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
