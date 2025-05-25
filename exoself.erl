-module(exoself).
-compile(export_all).
-include("records.hrl").
% -record(state,{file_name,genotype,idsNpids,cx_pid,spids,npids,apids,highest_fitness,tot_evaluations,tot_cycles}).
-define(MAX_ATTEMPTS,50).

map()-> map(ffnn).
map(FileName)->
	Genotype=genotype:load_from_file(FileName),
	spawn(exoself,prep,[FileName,Genotype]).

prep(FileName,Genotype)->
  io:format("ExoSelf: Starting prep for ~p~n", [FileName]),
	rand:seed(exsplus),
	IdsNPIds = ets:new(idsNpids,[set,private]), 
	Cx = genotype:read(Genotype,cortex),
	Sensor_Ids = Cx#cortex.sensor_ids,
	Actuator_Ids = Cx#cortex.actuator_ids,
	NIds = Cx#cortex.neuron_ids,
	ScapePIds = spawn_Scapes(IdsNPIds,Genotype,Sensor_Ids,Actuator_Ids),
	spawn_CerebralUnits(IdsNPIds,cortex,[Cx#cortex.id]),
	spawn_CerebralUnits(IdsNPIds,sensor,Sensor_Ids),
	spawn_CerebralUnits(IdsNPIds,actuator,Actuator_Ids),
	spawn_CerebralUnits(IdsNPIds,neuron,NIds),
	link_Sensors(Genotype,Sensor_Ids,IdsNPIds),
	link_Actuators(Genotype,Actuator_Ids,IdsNPIds),
	link_Neurons(Genotype,NIds,IdsNPIds),
	{SPIds,NPIds,APIds}=link_Cortex(Cx,IdsNPIds),
	Cx_PId = ets:lookup_element(IdsNPIds,Cx#cortex.id,2),
  io:format("ExoSelf: Entering main loop~n"),
	loop(FileName,Genotype,IdsNPIds,Cx_PId,SPIds,NPIds,APIds,ScapePIds,0,0,0,0,1).

loop(FileName,Genotype,IdsNPIds,Cx_PId,SPIds,NPIds,APIds,ScapePIds,HighestFitness,EvalAcc,CycleAcc,TimeAcc,Attempt)->
  io:format("ExoSelf: Waiting for evaluation (Attempt ~p)~n", [Attempt]),
	receive
		{Cx_PId,evaluation_completed,Fitness,Cycles,Time}->
			{U_HighestFitness,U_Attempt}=case Fitness > HighestFitness of
				true ->
					[NPId ! {self(),weight_backup} || NPId <- NPIds],
					{Fitness,0};
				false ->
					Perturbed_NPIds=get(perturbed),
					[NPId ! {self(),weight_restore} || NPId <- Perturbed_NPIds],
					{HighestFitness,Attempt+1}
			end,
			case U_Attempt >= ?MAX_ATTEMPTS of
				true ->	%End training
					U_CycleAcc = CycleAcc+Cycles,
					U_TimeAcc = TimeAcc+Time,
					backup_genotype(FileName,IdsNPIds,Genotype,NPIds),
					terminate_phenotype(Cx_PId,SPIds,NPIds,APIds,ScapePIds),
					io:format("Cortex:~p finished training. Genotype has been backed up.~n Fitness:~p~n TotEvaluations:~p~n TotCycles:~p~n TimeAcc:~p~n",
						[Cx_PId,U_HighestFitness,EvalAcc,U_CycleAcc,U_TimeAcc]),
					
					case whereis(trainer) of
						undefined ->
							ok;
						PId -> 
							PId ! {self(),U_HighestFitness,EvalAcc,U_CycleAcc,U_TimeAcc}
					end;
				false -> %Continue training
					Tot_Neurons = length(NPIds),
					MP = 1/math:sqrt(Tot_Neurons),
					Perturb_NPIds=[NPId || NPId <- NPIds, rand:uniform()<MP],
					put(perturbed,Perturb_NPIds),
					[NPId ! {self(),weight_perturb} || NPId <- Perturb_NPIds],
					Cx_PId ! {self(),reactivate},
					loop(FileName,Genotype,IdsNPIds,Cx_PId,SPIds,NPIds,APIds,ScapePIds,U_HighestFitness,EvalAcc+1,CycleAcc+Cycles,TimeAcc+Time,U_Attempt)
			end
	end.

	spawn_CerebralUnits(IdsNPIds,CerebralUnitType,[Id|Ids])-> 
		PId = CerebralUnitType:gen(self(),node()),
		ets:insert(IdsNPIds,{Id,PId}), 
		ets:insert(IdsNPIds,{PId,Id}), 
		spawn_CerebralUnits(IdsNPIds,CerebralUnitType,Ids); 
	spawn_CerebralUnits(_IdsNPIds,_CerebralUnitType,[])-> 
		true.

	spawn_Scapes(IdsNPIds,Genotype,Sensor_Ids,Actuator_Ids)-> 
		Sensor_Scapes = [(genotype:read(Genotype,Id))#sensor.scape || Id<-Sensor_Ids], 
		Actuator_Scapes = [(genotype:read(Genotype,Id))#actuator.scape || Id<-Actuator_Ids], 
		Unique_Scapes = Sensor_Scapes++(Actuator_Scapes--Sensor_Scapes), 
		SN_Tuples=[{scape:gen(self(),node()),ScapeName} || {private,ScapeName}<-Unique_Scapes], 
		[ets:insert(IdsNPIds,{ScapeName,PId}) || {PId,ScapeName} <- SN_Tuples], 
		[ets:insert(IdsNPIds,{PId,ScapeName}) || {PId,ScapeName} <-SN_Tuples], 
		[PId ! {self(),ScapeName} || {PId,ScapeName} <- SN_Tuples],
		[PId || {PId,_ScapeName} <-SN_Tuples].

  link_Sensors(Genotype,[SId|Sensor_Ids],IdsNPIds) ->
    R=genotype:read(Genotype,SId),
    SPId = ets:lookup_element(IdsNPIds,SId,2),
    Cx_PId = ets:lookup_element(IdsNPIds,R#sensor.cortex_id, 2),
    SName = R#sensor.name,
    Fanout_Ids = R#sensor.fanout_ids,
    Fanout_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- Fanout_Ids],
    Scape=case R#sensor.scape of
        {private,ScapeName}->
            ets:lookup_element(IdsNPIds,ScapeName,2)
    end,
    SPId ! {self(),{SId,Cx_PId,Scape,SName,R#sensor.vector_length, Fanout_PIds}},
		link_Sensors(Genotype,Sensor_Ids,IdsNPIds);
	link_Sensors(_Genotype,[],_IdsNPIds)->
		ok.


	link_Actuators(Genotype,[AId|Actuator_Ids],IdsNPIds) ->
		R=genotype:read(Genotype,AId),
		APId = ets:lookup_element(IdsNPIds,AId,2),
		Cx_PId = ets:lookup_element(IdsNPIds,R#actuator.cortex_id, 2),
		AName = R#actuator.name,
		Fanin_Ids = R#actuator.fanin_ids,
		Fanin_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- Fanin_Ids],
		Scape=case R#actuator.scape of
			{private,ScapeName}->
				ets:lookup_element(IdsNPIds,ScapeName,2)
		end,
		APId ! {self(),{AId,Cx_PId,Scape,AName,Fanin_PIds}},
		link_Actuators(Genotype,Actuator_Ids,IdsNPIds);
	link_Actuators(_Genotype,[],_IdsNPIds)->
		ok.
	link_Neurons(Genotype,[NId|Neuron_Ids],IdsNPIds) ->
		R=genotype:read(Genotype,NId),
		NPId = ets:lookup_element(IdsNPIds,NId,2),
		Cx_PId = ets:lookup_element(IdsNPIds,R#neuron.cortex_id,2),
		AFName = R#neuron.activation_function,
		Input_IdPs = R#neuron.input_ids,
		Output_Ids = R#neuron.output_ids,
		Input_PIdPs = convert_neuron_weights_to_process_weights(IdsNPIds,Input_IdPs,[]),
		Output_PIds = [ets:lookup_element(IdsNPIds,Id,2) || Id <- Output_Ids],
		NPId ! {self(),{NId,Cx_PId,AFName,Input_PIdPs,Output_PIds}},
		link_Neurons(Genotype,Neuron_Ids,IdsNPIds);
	link_Neurons(_Genotype,[],_IdsNPIds)->
		ok.

	link_Cortex(Cx,IdsNPIds) ->
		Cx_Id = Cx#cortex.id,
		Cx_PId = ets:lookup_element(IdsNPIds,Cx_Id,2),
		SIds = Cx#cortex.sensor_ids,
		AIds = Cx#cortex.actuator_ids,
		NIds = Cx#cortex.neuron_ids,
		SPIds = [ets:lookup_element(IdsNPIds,SId,2) || SId <- SIds],
		NPIds = [ets:lookup_element(IdsNPIds,NId,2) || NId <- NIds],
		APIds = [ets:lookup_element(IdsNPIds,AId,2) || AId <- AIds],
		Cx_PId ! {self(),Cx_Id,SPIds,NPIds,APIds},
		{SPIds,NPIds,APIds}.

backup_genotype(FileName,IdsNPIds,Genotype,NPIds)->
	Neuron_IdsNWeights = get_backup(NPIds,[]),
	update_genotype(IdsNPIds,Genotype,Neuron_IdsNWeights),
	genotype:save_to_file(Genotype,FileName),
	io:format("Finished updating genotype to file:~p~n",[FileName]).

	get_backup([NPId|NPIds],Acc)->
		NPId ! {self(),get_backup},
		receive
			{NPId,NId,WeightTuples}->
				get_backup(NPIds,[{NId,WeightTuples}|Acc])
		end;
	get_backup([],Acc)->
		Acc.

	update_genotype(IdsNPIds,Genotype,[{N_Id,PIdPs}|WeightPs])->
		N = genotype:read(Genotype,N_Id),
		Updated_InputIdPs = convert_process_weights_to_neuron_weights(IdsNPIds,PIdPs,[]),
		U_N = N#neuron{input_ids = Updated_InputIdPs},
		genotype:write(Genotype,U_N),
		update_genotype(IdsNPIds,Genotype,WeightPs);
	update_genotype(_IdsNPIds,_Genotype,[])->
		ok.
	
convert_neuron_weights_to_process_weights(_IdsNPIds,[{bias,Bias}],Acc)->
    lists:reverse([Bias|Acc]);
convert_neuron_weights_to_process_weights(IdsNPIds,[{Id,Weights}|Fanin_IdPs],Acc)->
    convert_neuron_weights_to_process_weights(IdsNPIds,Fanin_IdPs,
        [{ets:lookup_element(IdsNPIds,Id,2),Weights}|Acc]).
convert_process_weights_to_neuron_weights(IdsNPIds,[{PId,Weights}|Input_PIdPs],Acc) ->
    convert_process_weights_to_neuron_weights(
        IdsNPIds,
        Input_PIdPs,
        [{ets:lookup_element(IdsNPIds,PId,2),Weights}|Acc]
    );
convert_process_weights_to_neuron_weights(_IdsNPIds,[Bias],Acc) ->
    lists:reverse([{bias,Bias}|Acc]).

terminate_phenotype(Cx_PId,SPIds,NPIds,APIds,ScapePIds)->
  [PId ! {self(),terminate} || PId <- SPIds],
  [PId ! {self(),terminate} || PId <- APIds],
  [PId ! {self(),terminate} || PId <- NPIds],
  [PId ! {self(),terminate} || PId <- ScapePIds],
  Cx_PId ! {self(),terminate}.
