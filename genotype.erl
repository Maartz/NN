-module(genotype).
-compile(export_all).
-include("records.hrl").

construct(Morphology,HiddenLayerDensities) ->
	construct(ffnn,Morphology,HiddenLayerDensities).
construct(FileName,Morphology,HiddenLayerDensities) ->
    rand:seed(exsss),
	S = morphology:get_InitSensor(Morphology),
	A = morphology:get_InitActuator(Morphology),
	Output_VL = A#actuator.vector_length,
	LayerDensities = lists:append(HiddenLayerDensities,[Output_VL]),
	Cx_Id = cortex,
	
	Neurons = create_NeuroLayers(Cx_Id,S,A,LayerDensities), 
	[Input_Layer|_] = Neurons, 
	[Output_Layer|_] = lists:reverse(Neurons), 
	FL_NIds = [N#neuron.id || N <- Input_Layer], 
	LL_NIds = [N#neuron.id || N <-  Output_Layer], 
	NIds = [N#neuron.id || N <- lists:flatten(Neurons)],
	Sensor = S#sensor{cortex_id = Cx_Id,fanout_ids = FL_NIds},
	Actuator = A#actuator{cortex_id = Cx_Id,fanin_ids = LL_NIds},
	Cortex = create_Cortex(Cx_Id,[S#sensor.id],[A#actuator.id],NIds), 
	Genotype = lists:flatten([Cortex,Sensor,Actuator,Neurons]),
	save_genotype(FileName,Genotype),
	Genotype.

create_NeuroLayers(Cx_Id,Sensor,Actuator,LayerDensities) ->
    Input_IdPs = [{Sensor#sensor.id,Sensor#sensor.vector_length}],
    Tot_Layers = length(LayerDensities),
    [FL_Neurons|Next_LDs] = LayerDensities, 
    NIds = [{neuron,{1,Id}}|| Id <- helpers:generate_ids(FL_Neurons,[])],
    create_NeuroLayers(Cx_Id,Actuator#actuator.id,1,Tot_Layers,Input_IdPs,NIds,Next_LDs,[]). 

create_NeuroLayers(Cx_Id,Actuator_Id,LayerIndex,Tot_Layers,Input_IdPs,NIds,[Next_LD|LDs],Acc) ->
    Output_NIds = [{neuron,{LayerIndex+1,Id}} || Id <- helpers:generate_ids(Next_LD,[])], 
    Layer_Neurons = create_NeuroLayer(Cx_Id,Input_IdPs,NIds,Output_NIds,[]), 
    Next_InputIdPs = [{NId,1}|| NId <- NIds],
    create_NeuroLayers(Cx_Id,Actuator_Id,LayerIndex+1,Tot_Layers,Next_InputIdPs,Output_NIds,LDs,[Layer_Neurons|Acc]);
create_NeuroLayers(Cx_Id,Actuator_Id,Tot_Layers,Tot_Layers,Input_IdPs,NIds,[],Acc) -> 
    Output_Ids = [Actuator_Id], 
    Layer_Neurons = create_NeuroLayer(Cx_Id,Input_IdPs,NIds,Output_Ids,[]), 
    lists:reverse([Layer_Neurons|Acc]).

create_NeuroLayer(Cx_Id,Input_IdPs,[Id|NIds],Output_Ids,Acc) ->
    Neuron = create_Neuron(Input_IdPs,Id,Cx_Id,Output_Ids), 
    create_NeuroLayer(Cx_Id,Input_IdPs,NIds,Output_Ids,[Neuron|Acc]); 
create_NeuroLayer(_Cx_Id,_Input_IdPs,[],_Output_Ids,Acc) ->
    Acc.

create_Neuron(Input_IdPs,Id,Cx_Id,Output_Ids)-> 
    Proper_InputIdPs = create_NeuralInput(Input_IdPs,[]), 
    #neuron{id=Id, cortex_id = Cx_Id,activation_function = tanh, input_ids = Proper_InputIdPs,output_ids=Output_Ids}. 

create_NeuralInput([{Input_Id,Input_VL}|Input_IdPs],Acc) ->
    Weights = create_NeuralWeights(Input_VL,[]),
    create_NeuralInput(Input_IdPs,[{Input_Id,Weights}|Acc]); 
create_NeuralInput([],Acc)-> 
    lists:reverse([{bias,rand:uniform()-0.5}|Acc]).
			 
create_NeuralWeights(0,Acc) ->
    Acc; 
create_NeuralWeights(Index,Acc) ->
    W = rand:uniform()-0.5, 
    create_NeuralWeights(Index-1,[W|Acc]). 

create_Cortex(Cx_Id,S_Ids,A_Ids,NIds) ->
    #cortex{id = Cx_Id, sensor_ids=S_Ids, actuator_ids=A_Ids, neuron_ids = NIds}.

save_genotype(FileName,Genotype)->
	TId = ets:new(FileName, [public,set,{keypos,2}]),
	[ets:insert(TId,Element) || Element <- Genotype],
	ets:tab2file(TId,FileName).
		
save_to_file(Genotype,FileName)->
	ets:tab2file(Genotype,FileName).
	
load_from_file(FileName)->
	{ok,TId} = ets:file2tab(FileName),
	TId.

read(TId,Key)->
	[R] = ets:lookup(TId,Key),
	R.

write(TId,R)->
	ets:insert(TId,R).
	
print(FileName)->
	Genotype = load_from_file(FileName),
	Cx = read(Genotype,cortex),
	SIds = Cx#cortex.sensor_ids,
	NIds = Cx#cortex.neuron_ids,
	AIds = Cx#cortex.actuator_ids,
	io:format("~p~n",[Cx]),
	[io:format("~p~n",[read(Genotype,Id)]) || Id <- SIds],
	[io:format("~p~n",[read(Genotype,Id)]) || Id <- NIds],
	[io:format("~p~n",[read(Genotype,Id)]) || Id <- AIds].