%%% # Genotype Module%%%
%%% The genotype module handles the **specification and persistence** of neural
%%% network structures. It creates, stores, loads, and manipulates network
%%% blueprints independent of their running phenotype instances.
%%%
%%% ## Genotype vs Phenotype
%%%
%%% - **Genotype**: Network specification stored in ETS tables and files
%%%   - Defines structure: sensors, neurons, actuators, connections
%%%   - Stores weights: synaptic connection strengths
%%%   - Persisted to disk between training sessions
%%%   - Modified by evolutionary operators (mutation, crossover)
%%%
%%% - **Phenotype**: Running network of processes (see exoself.erl)
%%%   - Spawned from genotype specification
%%%   - Processes communicate via messages
%%%   - Terminated after training
%%%   - Trained weights saved back to genotype
%%%
%%% ## Data Storage
%%%
%%% Genotypes are stored in two ways:
%%% 1. **Mnesia database**: For multi-agent evolutionary systems
%%% 2. **ETS table files**: For standalone networks (XOR example)
%%%
%%% ## Key Operations
%%%
%%% - **construct/2,3**: Create new network from morphology and layer densities
%%% - **save_to_file/2**: Persist genotype to disk
%%% - **load_from_file/1**: Load genotype from disk
%%% - **clone_Agent/1,2**: Create genetic copy of agent
%%% - **delete_Agent/1,2**: Remove agent from database
%%% - **print/1**: Debug print network structure
%%%
%%% ## Network Construction
%%%
%%% Networks are built layer-by-layer:
%%% 1. Create sensors (input layer)
%%% 2. Create hidden layer neurons with random weights
%%% 3. Create output layer neurons
%%% 4. Create actuators (output layer)
%%% 5. Create cortex (coordinator)
%%% 6. Wire connections between layers
%%%
%%% ### Example Network Construction
%%% ```
%%% % Create XOR network with 3 hidden neurons
%%% genotype:construct(my_network, xor_mimic, [3]).
%%%
%%% % Load and run it
%%% exoself:map(my_network).
%%% '''
%%%
%%% ## File Format
%%%
%%% Genotype files are ETS table dumps containing:
%%% - 1 cortex record
%%% - N sensor records
%%% - M neuron records (with weights)
%%% - K actuator records
%%%
%%% Files are binary, created by `ets:tab2file/2`.

-module(genotype).
-compile(export_all).
-include("records.hrl").

-export([
    construct/2,
    construct/3,
    load_from_file/1,
    save_to_file/2,
    save_genotype/2,
    print/1,
    read/1,
    write/1,
    delete/1,
    delete_Agent/1,
    delete_Agent/2,
    clone_Agent/1,
    clone_Agent/2,
    test/0,
    create_test/0
]).

%%==============================================================================
%% Utility Functions
%%==============================================================================

%% @private
%% Wrapper around helpers:generate_id/0
generate_UniqueId() ->
    helpers:generate_id().

%%==============================================================================
%% Agent Construction (Evolutionary System)
%%==============================================================================

%% @private
%% Construct an agent for evolutionary population
%%
%% Creates a complete agent with genotype stored in Mnesia.
%% Used by the evolutionary/population system, not for standalone networks.
construct_Agent(SpecieId, AgentId, SpecCon) ->
  rand:seed(exsplus),
  Generation = 0,
  {Cx_Id, Pattern} = construct_Cortex(AgentId, Generation, SpecCon),
  Agent = #agent{
             id = AgentId,
             cortex_id = Cx_Id,
             specie_id = SpecieId,
             constraint = SpecCon,
             generation = Generation,
             pattern = Pattern,
             evolution_history = []
            },
  write(Agent),
  update_fingerprint(AgentId).

%% @private
construct_Cortex(AgentId, Generation, SpecCon) ->
  Cx_Id = {{origin, generate_UniqueId()}, cortex},
  Morphology = SpecCon#constraint.morphology,
  Sensors = [S#sensor{id = {{-1, generate_UniqueId()}, sensor}, cortex_id = Cx_Id} || S <- morphology:get_InitSensors(Morphology)],
  Actuators = [A#actuator{id = {{1, generate_UniqueId()}, actuator}, cortex_id = Cx_Id} || A <- morphology:get_InitActuators(Morphology)],
  N_IDs = construct_InitialNeuroLayer(Cx_Id, Generation, SpecCon, Sensors, Actuators, [], []),
  S_IDs = [S#sensor.id || S <- Sensors],
  A_IDs = [A#actuator.id || A <- Actuators],
  Cortex = #cortex{
              id = Cx_Id,
              agent_id = AgentId,
              neuron_ids = N_IDs,
              actuator_ids = A_IDs,
              sensor_ids = S_IDs
             },
  write(Cortex),
  {Cx_Id, [{0, N_IDs}]}.

%% @private
construct_InitialNeuroLayer(Cx_Id,Generation,SpecCon,Sensors,[A|Actuators],AAcc,NIdAcc)->
  N_IDs = [{{0, Unique_Id}, neuron} || Unique_Id <- helpers:generate_ids(A#actuator.vector_length, [])],
  U_Sensors = construct_InitialNeurons(Cx_Id, Generation, SpecCon, N_IDs, Sensors, A),
  U_A = A#actuator{fanin_ids = N_IDs},
  construct_InitialNeuroLayer(Cx_Id, Generation, SpecCon, U_Sensors, Actuators, [U_A | AAcc], lists:append(N_IDs, NIdAcc));
construct_InitialNeuroLayer(_Cx_Id, _Generation, _SpecCon, Sensors, [], AAcc, NIdAcc) ->
  [write(S) || S <- Sensors],
  [write(A) || A <- AAcc],
  NIdAcc.

%% @private
construct_InitialNeurons(Cx_Id, Generation, SpecCon, [N_Id | N_IDs], Sensors, Actuator) ->
  case rand:uniform() >= 0.5 of
    true ->
      S = lists:nth(rand:uniform(length(Sensors)), Sensors),
      U_Sensors = lists:keyreplace(S#sensor.id, 2, Sensors, S#sensor{fanout_ids=[N_Id | S#sensor.fanout_ids]}),
      Input_Specs = [{S#sensor.id, S#sensor.vector_length}];
    false ->
      U_Sensors = [S#sensor{fanout_ids = [N_Id | S#sensor.fanout_ids]} || S <- Sensors],
      Input_Specs = [{S#sensor.id, S#sensor.vector_length} || S <- Sensors]
  end,
  construct_Neuron(Cx_Id, Generation, SpecCon, N_Id, Input_Specs, [Actuator#actuator.id]),
  construct_InitialNeurons(Cx_Id, Generation, SpecCon, N_IDs, U_Sensors, Actuator);
construct_InitialNeurons(_Cx_Id, _Generation, _SpecCon, [], Sensors, Actuator) ->
  Sensors.

%% @private
construct_Neuron(Cx_Id, Generation, SpecCon, N_Id, Input_Specs, Output_IDs) ->
  Input_IdPs = create_InputIdPs(Input_Specs, []),
  Neuron = #neuron{
    id = N_Id,
    cortex_id = Cx_Id,
    generation = Generation,
    activation_function = generate_NeuronAF(SpecCon#constraint.neural_afs),
    input_ids = Input_IdPs,
    output_ids = Output_IDs,
    ro_ids = calculate_ROIds(N_Id, Output_IDs, [])
   },
  write(Neuron).

%% @private
create_InputIdPs([{Input_Id, Input_VL} | Input_IdPs], Acc) ->
  Weights = create_NeuralWeights(Input_VL, []),
  create_InputIdPs(Input_IdPs, [{Input_Id, Weights} | Acc]);
create_InputIdPs([], Acc) ->
  Acc.

%% @private
generate_NeuronAF(Activation_Functions) ->
  case Activation_Functions of
    [] ->
      tanh;
    Other ->
      lists:nth(rand:uniform(length(Other)), Other)
  end.

%% @private
calculate_ROIds(Self_Id, [Output_Id | Ids], Acc) ->
  case Output_Id of
    {_, actuator} ->
      calculate_ROIds(Self_Id, Ids, Acc);
    Output_Id ->
      {{TLI, _}, _} = Self_Id,
      {{LI, _}, _} = Output_Id,
      case LI =< TLI of
        true -> calculate_ROIds(Self_Id, Ids, [Output_Id | Acc]);
        false -> calculate_ROIds(Self_Id, Ids, Acc)
      end
  end;
calculate_ROIds(_Self_Id, [], Acc) ->
  lists:reverse(Acc).

%% @private
update_fingerprint(Agent_Id) ->
  A = read({agent, Agent_Id}),
  Cx = read({cortex, A#agent.cortex_id}),
  GeneralizedSensors = [(read({sensor, S_Id}))#sensor{id = undefined, cortex_id = undefined} || S_Id <- Cx#cortex.sensor_ids],
  GeneralizedActuators = [(read({actuator, A_Id}))#actuator{id = undefined, cortex_id = undefined} || A_Id <- Cx#cortex.actuator_ids],
  GeneralizedPattern = [{LayerIndex, length(LNIds)} || {LayerIndex, LNIds } <- A#agent.pattern],
  GeneralizedEvolutionHistory = generalize_EvoHist(A#agent.evolution_history, []),
  Fingerprint = {GeneralizedPattern, GeneralizedEvolutionHistory, GeneralizedSensors, GeneralizedActuators},
  write(A#agent{fingerprint = Fingerprint}).

%% Generalize evolutionary history by removing unique IDs from component tuples.
%% This allows structural comparison of networks independent of their specific IDs.
%% Handles three types of mutation records: 3-component, 2-component, and 1-component.

generalize_EvoHist([MutationRecord | Rest], Acc) ->
  GeneralizedRecord = generalize_mutation_record(MutationRecord),
  generalize_EvoHist(Rest, [GeneralizedRecord | Acc]);
generalize_EvoHist([], Acc) ->
  lists:reverse(Acc).

%% Generalize a mutation record with 3 components (e.g., add_neuron connecting 3 elements)
generalize_mutation_record({MutationOp, ComponentA, ComponentB, ComponentC}) ->
  {MutationOp,
   remove_unique_id(ComponentA),
   remove_unique_id(ComponentB),
   remove_unique_id(ComponentC)};

%% Generalize a mutation record with 2 components (e.g., add_link between 2 neurons)
generalize_mutation_record({MutationOp, ComponentA, ComponentB}) ->
  {MutationOp,
   remove_unique_id(ComponentA),
   remove_unique_id(ComponentB)};

%% Generalize a mutation record with 1 component (e.g., mutate_weights on single neuron)
generalize_mutation_record({MutationOp, ComponentA}) ->
  {MutationOp,
   remove_unique_id(ComponentA)}.

%% Remove unique ID from a component, keeping only layer index and type.
%% Transforms {{LayerIndex, UniqueId}, Type} -> {LayerIndex, Type}
remove_unique_id({{LayerIndex, _UniqueId}, ComponentType}) ->
  {LayerIndex, ComponentType}.

%%==============================================================================
%% Mnesia I/O Operations
%%==============================================================================

%% @doc Read a record from Mnesia database
%%
%% Reads a single record from the Mnesia database. Returns `undefined'
%% if the record doesn't exist.
%%
%% === Parameters ===
%% - `TnK' - Tuple `{TableName, Key}' for record lookup
%%
%% === Returns ===
%% The record if found, or `undefined' if not found
%%
%% === Examples ===
%% ```
%% Agent = genotype:read({agent, test}).
%% Cortex = genotype:read({cortex, cortex}).
%% '''
-spec read({atom(), term()}) -> term() | undefined.
read(TnK) ->
  case mnesia:read(TnK) of
    [] ->
      undefined;
    [R] ->
      R
  end.

%% @doc Read a record from Mnesia (dirty, no transaction)
%%
%% Faster than `read/1' but not transactionally safe.
%% Use for read-only operations outside transactions.
-spec dirty_read({atom(), term()}) -> term() | undefined.
dirty_read(TnK) ->
  case mnesia:dirty_read(TnK) of
    [] ->
      undefined;
    [R] ->
      R
  end.

%% @doc Write a record to Mnesia database
%%
%% Writes a record to the Mnesia database. Must be called within
%% a Mnesia transaction.
%%
%% === Parameters ===
%% - `R' - Record to write (sensor, neuron, actuator, cortex, agent, etc.)
%%
%% === Examples ===
%% ```
%% F = fun() -> genotype:write(#sensor{id=sensor1, ...}) end,
%% mnesia:transaction(F).
%% '''
-spec write(tuple()) -> ok.
write(R) ->
  mnesia:write(R).

%% @doc Delete a record from Mnesia database
%%
%% Deletes a record from the Mnesia database. Must be called within
%% a Mnesia transaction.
%%
%% === Parameters ===
%% - `TnK' - Tuple `{TableName, Key}' for record to delete
%%
%% === Examples ===
%% ```
%% F = fun() -> genotype:delete({agent, test}) end,
%% mnesia:transaction(F).
%% '''
-spec delete({atom(), term()}) -> ok.
delete(TnK) ->
  mnesia:delete(TnK).

%%==============================================================================
%% Agent Management
%%==============================================================================

%% @doc Delete an agent and all its components
%%
%% Removes an agent from the Mnesia database along with all its
%% cerebral units (cortex, sensors, neurons, actuators).
%%
%% **Warning**: This does not update the specie's agent list.
%% Use `delete_Agent/2' with `safe' option for proper cleanup.
%%
%% === Parameters ===
%% - `Agent_Id' - ID of the agent to delete
%%
%% === Examples ===
%% ```
%% F = fun() -> genotype:delete_Agent(test) end,
%% mnesia:transaction(F).
%% '''
-spec delete_Agent(term()) -> ok.
delete_Agent(Agent_Id)->
	A = read({agent,Agent_Id}),
	Cx = read({cortex,A#agent.cortex_id}),
	[delete({neuron,Id}) || Id <- Cx#cortex.neuron_ids],
	[delete({sensor,Id}) || Id <- Cx#cortex.sensor_ids],
	[delete({actuator,Id}) || Id <- Cx#cortex.actuator_ids],
	delete({cortex,A#agent.cortex_id}),
	delete({agent,Agent_Id}).

%% @doc Safely delete an agent (updates specie records)
%%
%% Deletes an agent and removes it from its specie's agent list.
%% This is the recommended way to delete agents in evolutionary systems.
%%
%% === Parameters ===
%% - `Agent_Id' - ID of the agent to delete
%% - `safe' - Atom indicating safe deletion mode
%%
%% === Examples ===
%% ```
%% genotype:delete_Agent(agent_001, safe).
%% '''
-spec delete_Agent(term(), safe) -> ok.
delete_Agent(Agent_Id,safe)->
	F = fun()->
		A = genotype:read({agent,Agent_Id}),
		S = genotype:read({specie,A#agent.specie_id}),
		Agent_Ids = S#specie.agent_ids,
		write(S#specie{agent_ids = lists:delete(Agent_Id,Agent_Ids)}),
		delete_Agent(Agent_Id)
	end,
	Result=mnesia:transaction(F),
	io:format("delete_agent(Agent_Id,safe):~p Result:~p~n",[Agent_Id,Result]).

%% @doc Clone an agent (generate new ID automatically)
%%
%% Creates a genetic copy of an agent with a new unique ID.
%% The clone has identical structure and weights but is a separate entity.
%%
%% === Parameters ===
%% - `Agent_Id' - ID of the agent to clone
%%
%% === Returns ===
%% ID of the newly created clone
%%
%% === Examples ===
%% ```
%% CloneId = genotype:clone_Agent(test).
%% '''
-spec clone_Agent(term()) -> {float(), agent}.
clone_Agent(Agent_Id)->
	CloneAgent_Id = {helpers:generate_id(),agent},
	clone_Agent(Agent_Id,CloneAgent_Id).

%% @doc Clone an agent with specified ID
%%
%% Creates a genetic copy of an agent using the provided clone ID.
%% Used internally and by evolutionary operators.
%%
%% === Parameters ===
%% - `Agent_Id' - ID of the agent to clone
%% - `CloneAgent_Id' - ID to assign to the clone
%%
%% === Returns ===
%% ID of the newly created clone
-spec clone_Agent(term(), term()) -> term().
clone_Agent(Agent_Id,CloneAgent_Id)->
	F = fun()->
		A = read({agent,Agent_Id}),
		Cx = read({cortex,A#agent.cortex_id}),
		IdsNCloneIds = ets:new(idsNcloneids,[set,private]),
		ets:insert(IdsNCloneIds,{bias,bias}),
		ets:insert(IdsNCloneIds,{Agent_Id,CloneAgent_Id}),
		[CloneCx_Id] = map_ids(IdsNCloneIds,[A#agent.cortex_id],[]),
		CloneN_Ids = map_ids(IdsNCloneIds,Cx#cortex.neuron_ids,[]),
		CloneS_Ids = map_ids(IdsNCloneIds,Cx#cortex.sensor_ids,[]),
		CloneA_Ids = map_ids(IdsNCloneIds,Cx#cortex.actuator_ids,[]),
		clone_neurons(IdsNCloneIds,Cx#cortex.neuron_ids),
		clone_sensors(IdsNCloneIds,Cx#cortex.sensor_ids),
		clone_actuators(IdsNCloneIds,Cx#cortex.actuator_ids),

		write(Cx#cortex{
			id = CloneCx_Id,
			agent_id = CloneAgent_Id,
			sensor_ids = CloneS_Ids,
			actuator_ids = CloneA_Ids,
			neuron_ids = CloneN_Ids
		}),
		write(A#agent{
			id = CloneAgent_Id,
			cortex_id = CloneCx_Id
		}),
		ets:delete(IdsNCloneIds)
	end,
	mnesia:transaction(F),
	CloneAgent_Id.

%% @private
	map_ids(TableName,[Id|Ids],Acc)->
		CloneId=case Id of
			{{LayerIndex,_NumId},Type}->
				{{LayerIndex, helpers:generate_id()},Type};
			{_NumId,Type}->
				{helpers:generate_id(),Type}
		end,
		ets:insert(TableName,{Id,CloneId}),
		map_ids(TableName,Ids,[CloneId|Acc]);
	map_ids(_TableName,[],Acc)->
		Acc.

%% @private
clone_sensors(TableName,[S_Id|S_Ids])->
	S = read({sensor,S_Id}),
	CloneS_Id = ets:lookup_element(TableName,S_Id,2),
	CloneCx_Id = ets:lookup_element(TableName,S#sensor.cortex_id,2),
	CloneFanout_Ids =[ets:lookup_element(TableName,Fanout_Id,2)|| Fanout_Id <- S#sensor.fanout_ids],
	write(S#sensor{
		id = CloneS_Id,
		cortex_id = CloneCx_Id,
		fanout_ids = CloneFanout_Ids
	}),
	clone_sensors(TableName,S_Ids);
clone_sensors(_TableName,[])->
  done.

%% @private
clone_actuators(TableName,[A_Id|A_Ids])->
	A = read({actuator,A_Id}),
	CloneA_Id = ets:lookup_element(TableName,A_Id,2),
	CloneCx_Id = ets:lookup_element(TableName,A#actuator.cortex_id,2),
	CloneFanin_Ids =[ets:lookup_element(TableName,Fanin_Id,2)|| Fanin_Id <- A#actuator.fanin_ids],
	write(A#actuator{
		id = CloneA_Id,
		cortex_id = CloneCx_Id,
		fanin_ids = CloneFanin_Ids
	}),
	clone_actuators(TableName,A_Ids);
clone_actuators(_TableName,[])->
  done.

%% @private
clone_neurons(TableName,[N_Id|N_Ids])->
  N = read({neuron,N_Id}),
  CloneN_Id = ets:lookup_element(TableName,N_Id,2),
  CloneCx_Id = ets:lookup_element(TableName,N#neuron.cortex_id,2),
  CloneInput_IdPs =  [{ets:lookup_element(TableName,I_Id,2),Weights}|| {I_Id,Weights} <- N#neuron.input_ids],
  CloneOutput_Ids = [ets:lookup_element(TableName,O_Id,2)|| O_Id <- N#neuron.output_ids],
  CloneRO_Ids =[ets:lookup_element(TableName,RO_Id,2)|| RO_Id <- N#neuron.ro_ids],
  write(N#neuron{
    id = CloneN_Id,
    cortex_id = CloneCx_Id,
    input_ids = CloneInput_IdPs,
    output_ids = CloneOutput_Ids,
    ro_ids = CloneRO_Ids
  }),
  clone_neurons(TableName,N_Ids);
clone_neurons(_TableName,[])->
  done.

%% @private
speciate(Agent_Id)->
	update_fingerprint(Agent_Id),
	A = read({agent,Agent_Id}),
	case A#agent.id of
		test ->
			write(A#agent{fitness = undefined});
		_ ->
			Parent_S = read({specie,A#agent.specie_id}),
			P = read({population,Parent_S#specie.population_id}),
			case [Id || Id <- P#population.specie_ids, (read({specie,Id}))#specie.fingerprint == A#agent.fingerprint] of
				[] ->
					Specie_Id = population_monitor:create_specie(P#population.id,A#agent.constraint,A#agent.fingerprint),
					S = read({specie,Specie_Id}),
					U_A = A#agent{specie_id=Specie_Id,fitness = undefined},
					U_S = S#specie{agent_ids = [Agent_Id]},
					write(U_A),
					write(U_S);
				[Specie_Id] ->
					S = read({specie,Specie_Id}),
					U_A = A#agent{specie_id=Specie_Id,fitness = undefined},
					U_S = S#specie{agent_ids = [Agent_Id|S#specie.agent_ids]},
					write(U_A),
					write(U_S)
			end
	end.

%%==============================================================================
%% Testing Functions
%%==============================================================================

%% @doc Test function for genotype operations
%%
%% Creates a test agent, clones it, prints both, and deletes them.
%% Useful for verifying the genotype system is working correctly.
%%
%% === Examples ===
%% ```
%% genotype:test().
%% '''
-spec test() -> {atomic, ok} | {aborted, term()}.
test()->
	Specie_Id = test,
	Agent_Id = test,
	CloneAgent_Id = test_clone,
	SpecCon = #constraint{},
	F = fun()->
		construct_Agent(Specie_Id,Agent_Id,SpecCon),
		clone_Agent(Specie_Id,CloneAgent_Id),
		print(Agent_Id),
		print(CloneAgent_Id),
		delete_Agent(Agent_Id),
		delete_Agent(CloneAgent_Id)
	end,
	mnesia:transaction(F).

%% @doc Create a test agent (persistent)
%%
%% Creates or recreates a test agent for interactive experimentation.
%% The agent persists in Mnesia until explicitly deleted.
%%
%% === Examples ===
%% ```
%% genotype:create_test().
%% exoself:map(test).
%% '''
-spec create_test() -> {atomic, ok} | {aborted, term()}.
create_test()->
	Specie_Id = test,
	Agent_Id = test,
	SpecCon = #constraint{},
	F = fun()->
		case genotype:read({agent,test}) of
			undefined ->
				construct_Agent(Specie_Id,Agent_Id,SpecCon),
				print(Agent_Id);
			_ ->
				delete_Agent(Agent_Id),
				construct_Agent(Specie_Id,Agent_Id,SpecCon),
				print(Agent_Id)
		end
	end,
	mnesia:transaction(F).

%%==============================================================================
%% Standalone Network Construction (ETS-based)
%%==============================================================================

%% @doc Construct a standalone neural network
%%
%% Creates a feedforward neural network with specified morphology
%% and hidden layer configuration. Saves to file as ETS table.
%%
%% This is the main entry point for creating standalone networks
%% for the XOR problem and similar tasks.
%%
%% === Parameters ===
%% - `Morphology' - Problem domain (e.g., `xor_mimic')
%% - `HiddenLayerDensities' - List of neuron counts per hidden layer
%%
%% === Returns ===
%% List of all genotype records (cortex, sensors, neurons, actuators)
%%
%% === Examples ===
%% ```
%% % Create XOR network with 3 hidden neurons
%% genotype:construct(xor_mimic, [3]).
%%
%% % Create network with 2 hidden layers (5 and 3 neurons)
%% genotype:construct(xor_mimic, [5, 3]).
%% '''
-spec construct(atom(), list(pos_integer())) -> list(tuple()).
construct(Morphology,HiddenLayerDensities) ->
	construct(ffnn,Morphology,HiddenLayerDensities).

%% @doc Construct a standalone neural network with custom filename
%%
%% Same as `construct/2' but allows specifying the output filename.
%%
%% === Parameters ===
%% - `FileName' - Atom to use as filename
%% - `Morphology' - Problem domain (e.g., `xor_mimic')
%% - `HiddenLayerDensities' - List of neuron counts per hidden layer
%%
%% === Returns ===
%% List of all genotype records
%%
%% === Examples ===
%% ```
%% genotype:construct(my_network, xor_mimic, [3]).
%% '''
-spec construct(atom(), atom(), list(pos_integer())) -> list(tuple()).
construct(FileName,Morphology,HiddenLayerDensities) ->
  rand:seed(exsplus),
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

%% @private
create_NeuroLayers(Cx_Id,Sensor,Actuator,LayerDensities) ->
    Input_IdPs = [{Sensor#sensor.id,Sensor#sensor.vector_length}],
    Tot_Layers = length(LayerDensities),
    [FL_Neurons|Next_LDs] = LayerDensities,
    NIds = [{neuron,{1,Id}}|| Id <- helpers:generate_ids(FL_Neurons,[])],
    create_NeuroLayers(Cx_Id,Actuator#actuator.id,1,Tot_Layers,Input_IdPs,NIds,Next_LDs,[]).

%% @private
create_NeuroLayers(Cx_Id,Actuator_Id,LayerIndex,Tot_Layers,Input_IdPs,NIds,[Next_LD|LDs],Acc) ->
    Output_NIds = [{neuron,{LayerIndex+1,Id}} || Id <- helpers:generate_ids(Next_LD,[])],
    Layer_Neurons = create_NeuroLayer(Cx_Id,Input_IdPs,NIds,Output_NIds,[]),
    Next_InputIdPs = [{NId,1}|| NId <- NIds],
    create_NeuroLayers(Cx_Id,Actuator_Id,LayerIndex+1,Tot_Layers,Next_InputIdPs,Output_NIds,LDs,[Layer_Neurons|Acc]);
create_NeuroLayers(Cx_Id,Actuator_Id,Tot_Layers,Tot_Layers,Input_IdPs,NIds,[],Acc) ->
    Output_Ids = [Actuator_Id],
    Layer_Neurons = create_NeuroLayer(Cx_Id,Input_IdPs,NIds,Output_Ids,[]),
    lists:reverse([Layer_Neurons|Acc]).

%% @private
create_NeuroLayer(Cx_Id,Input_IdPs,[Id|NIds],Output_Ids,Acc) ->
    Neuron = create_Neuron(Input_IdPs,Id,Cx_Id,Output_Ids),
    create_NeuroLayer(Cx_Id,Input_IdPs,NIds,Output_Ids,[Neuron|Acc]);
create_NeuroLayer(_Cx_Id,_Input_IdPs,[],_Output_Ids,Acc) ->
    Acc.

%% @private
create_Neuron(Input_IdPs,Id,Cx_Id,Output_Ids)->
    Proper_InputIdPs = create_NeuralInput(Input_IdPs,[]),
    #neuron{id=Id, cortex_id = Cx_Id,activation_function = tanh, input_ids = Proper_InputIdPs,output_ids=Output_Ids}.

%% @private
create_NeuralInput([{Input_Id,Input_VL}|Input_IdPs],Acc) ->
    Weights = create_NeuralWeights(Input_VL,[]),
    create_NeuralInput(Input_IdPs,[{Input_Id,Weights}|Acc]);
create_NeuralInput([],Acc)->
    lists:reverse([{bias,rand:uniform()-0.5}|Acc]).

%% @private
create_NeuralWeights(0,Acc) ->
    Acc;
create_NeuralWeights(Index,Acc) ->
    W = rand:uniform()-0.5,
    create_NeuralWeights(Index-1,[W|Acc]).

%% @private
create_Cortex(Cx_Id,S_Ids,A_Ids,NIds) ->
    #cortex{id = Cx_Id, sensor_ids=S_Ids, actuator_ids=A_Ids, neuron_ids = NIds}.

%%==============================================================================
%% File I/O (ETS-based genotypes)
%%==============================================================================

%% @doc Save genotype to file
%%
%% Creates an ETS table from the genotype records and saves it to disk.
%% The file is a binary ETS table dump readable by `load_from_file/1'.
%%
%% === Parameters ===
%% - `FileName' - Atom to use as filename (no extension)
%% - `Genotype' - List of genotype records
%%
%% === Examples ===
%% ```
%% Genotype = genotype:construct(xor_mimic, [3]),
%% genotype:save_genotype(my_network, Genotype).
%% '''
-spec save_genotype(atom(), list(tuple())) -> ok | {error, term()}.
save_genotype(FileName,Genotype)->
	TId = ets:new(FileName, [public,set,{keypos,2}]),
	[ets:insert(TId,Element) || Element <- Genotype],
	ets:tab2file(TId,FileName).

%% @doc Save ETS table to file
%%
%% Saves an already-existing ETS table to disk. Used by ExoSelf
%% to persist trained weights after training.
%%
%% === Parameters ===
%% - `Genotype' - ETS table ID
%% - `FileName' - Atom to use as filename
%%
%% === Examples ===
%% ```
%% Genotype = genotype:load_from_file(ffnn),
%% % ... modify genotype ...
%% genotype:save_to_file(Genotype, ffnn).
%% '''
-spec save_to_file(ets:tid(), atom()) -> ok | {error, term()}.
save_to_file(Genotype,FileName)->
	ets:tab2file(Genotype,FileName).

%% @doc Load genotype from file
%%
%% Loads a genotype ETS table from disk. Returns an ETS table ID
%% that can be used with `read/2' and `write/2'.
%%
%% === Parameters ===
%% - `FileName' - Atom filename (no extension)
%%
%% === Returns ===
%% ETS table ID containing the genotype
%%
%% === Examples ===
%% ```
%% Genotype = genotype:load_from_file(ffnn),
%% Cortex = genotype:read(Genotype, cortex).
%% '''
-spec load_from_file(atom()) -> ets:tid().
load_from_file(FileName)->
	{ok,TId} = ets:file2tab(FileName),
	TId.

%% @doc Read a record from ETS genotype table
%%
%% Reads a record from an ETS-based genotype (not Mnesia).
%%
%% === Parameters ===
%% - `TId' - ETS table ID
%% - `Key' - Record key (e.g., `cortex', `{sensor, Id}')
%%
%% === Returns ===
%% The requested record
%%
%% === Examples ===
%% ```
%% Genotype = genotype:load_from_file(ffnn),
%% Cortex = genotype:read(Genotype, cortex).
%% '''
-spec read(ets:tid(), term()) -> tuple().
read(TId,Key)->
	[R] = ets:lookup(TId,Key),
	R.

%% @doc Write a record to ETS genotype table
%%
%% Writes a record to an ETS-based genotype (not Mnesia).
%%
%% === Parameters ===
%% - `TId' - ETS table ID
%% - `R' - Record to write
%%
%% === Examples ===
%% ```
%% Genotype = genotype:load_from_file(ffnn),
%% UpdatedNeuron = Neuron#neuron{...},
%% genotype:write(Genotype, UpdatedNeuron).
%% '''
-spec write(ets:tid(), tuple()) -> true.
write(TId,R)->
	ets:insert(TId,R).

%%==============================================================================
%% Debug Functions
%%==============================================================================

%% @doc Print agent structure (Mnesia-based)
%%
%% Prints the complete structure of an agent from Mnesia database,
%% including cortex, sensors, neurons, and actuators.
%%
%% Useful for debugging and understanding network topology.
%%
%% === Parameters ===
%% - `Agent_Id' - ID of the agent to print
%%
%% === Examples ===
%% ```
%% genotype:create_test(),
%% genotype:print(test).
%% '''
-spec print(term()) -> {atomic, ok} | {aborted, term()}.
print(Agent_Id)->
  F = fun() ->
          A = read({agent, Agent_Id}),
          Cx = read({cortex, A#agent.cortex_id}),
          io:format("~p~n", [A]),
          io:format("~p~n", [Cx]),
          [io:format("~p~n", [read({sensor, Id})]) || Id <- Cx#cortex.sensor_ids],
          [io:format("~p~n", [read({neuron, Id})]) || Id <- Cx#cortex.neuron_ids],
          [io:format("~p~n", [read({actuator, Id})]) || Id <- Cx#cortex.actuator_ids]
      end,
  mnesia:transaction(F).
