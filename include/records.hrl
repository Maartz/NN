%%% # Neural Network Record Definitions
%%%
%%% This file defines the core data structures used throughout the
%%% Feed-Forward Neural Network (FFNN) implementation.
%%%
%%% Each neural network component (sensor, neuron, actuator, cortex) is
%%% represented as a record and runs as an independent concurrent process
%%% communicating via message passing.

%%% ## Type Definitions

%%% A unique identifier generated using `helpers:generate_id/0`
-type unique_id() :: float().

%%% Layer position in the network (-1=sensor, 0=hidden, 1=actuator)
-type layer_index() :: integer().

%%% Component identifier with layer index and unique ID
-type component_id() :: {layer_index(), unique_id()}.

-type sensor_id() :: {component_id(), sensor}.
-type neuron_id() :: {component_id(), neuron}.
-type actuator_id() :: {component_id(), actuator}.
-type cortex_id() :: {component_id(), cortex}.
-type agent_id() :: {unique_id(), agent} | atom().
-type specie_id() :: {unique_id(), specie} | atom().
-type population_id() :: {unique_id(), population} | atom().

%%% Scape identifier: private scapes are spawned per agent, public are shared
-type scape_name() :: {private | public, atom()}.

-type activation_function() :: tanh | cos | gauss | abs | sin | linear.
-type weight() :: float().
-type weights() :: [weight()].

%%% Input connection with associated weights
-type input_id_pair() :: {sensor_id() | neuron_id(), weights()}.

%%% Network morphology type (e.g., xor_mimic)
-type morphology() :: atom().

%%% ## Record Definitions

%%% ### Constraint Record
%%%
%%% Constraint - defines morphology and allowed activation functions
%%%
%%% **Fields:**
%%% - `morphology` - Network morphology type (e.g., xor_mimic)
%%% - `neural_afs` - List of allowed activation functions
-record(constraint, {
    morphology = xor_mimic :: morphology(),
    neural_afs = [tanh, cos, gauss, abs] :: [activation_function()]
}).

%%% ### Sensor Record
%%%
%%% Sensor component - provides input to the neural network
%%%
%%% **Fields:**
%%% - `id` - Unique sensor identifier
%%% - `name` - Sensor function name (e.g., xor_GetInput)
%%% - `cortex_id` - ID of the cortex managing this sensor
%%% - `scape` - Environment scape providing sensory input
%%% - `vector_length` - Number of values in the input vector
%%% - `fanout_ids` - List of neuron IDs receiving this sensor's output
%%% - `generation` - Generation when this sensor was created
-record(sensor, {
    id :: sensor_id(),
    name :: atom(),
    cortex_id :: cortex_id(),
    scape :: scape_name(),
    vector_length :: pos_integer(),
    fanout_ids = [] :: [neuron_id()],
    generation :: non_neg_integer()
}).

%%% ### Actuator Record
%%%
%%% Actuator component - produces output from the neural network
%%%
%%% **Fields:**
%%% - `id` - Unique actuator identifier
%%% - `name` - Actuator function name (e.g., xor_SendOutput)
%%% - `cortex_id` - ID of the cortex managing this actuator
%%% - `scape` - Environment scape receiving actuator output
%%% - `vector_length` - Number of values in the output vector
%%% - `fanin_ids` - List of neuron IDs sending input to this actuator
%%% - `generation` - Generation when this actuator was created
-record(actuator, {
    id :: actuator_id(),
    name :: atom(),
    cortex_id :: cortex_id(),
    scape :: scape_name(),
    vector_length :: pos_integer(),
    fanin_ids = [] :: [neuron_id()],
    generation :: non_neg_integer()
}).

%%% ### Neuron Record
%%%
%%% Neuron - core processing unit computing weighted sum + activation function
%%%
%%% **Fields:**
%%% - `id` - Unique neuron identifier with layer index
%%% - `generation` - Generation when this neuron was created
%%% - `cortex_id` - ID of the cortex managing this neuron
%%% - `activation_function` - Activation function (tanh, cos, gauss, abs)
%%% - `input_ids` - List of {InputId, Weights} pairs for incoming connections
%%% - `output_ids` - List of neuron/actuator IDs receiving this neuron's output
%%% - `ro_ids` - Recurrent output IDs (for recurrent connections)
-record(neuron, {
    id :: neuron_id(),
    generation :: non_neg_integer(),
    cortex_id :: cortex_id(),
    activation_function :: activation_function(),
    input_ids = [] :: [input_id_pair() | {bias, weight()}],
    output_ids = [] :: [neuron_id() | actuator_id()],
    ro_ids = [] :: [neuron_id()]
}).

%%% ### Cortex Record
%%%
%%% Cortex - network coordinator synchronizing sensor-neuron-actuator cycles
%%%
%%% **Fields:**
%%% - `id` - Unique cortex identifier
%%% - `agent_id` - ID of the agent owning this cortex
%%% - `neuron_ids` - List of all neuron IDs in the network
%%% - `sensor_ids` - List of all sensor IDs
%%% - `actuator_ids` - List of all actuator IDs
-record(cortex, {
    id :: cortex_id(),
    agent_id :: agent_id(),
    neuron_ids = [] :: [neuron_id()],
    sensor_ids = [] :: [sensor_id()],
    actuator_ids = [] :: [actuator_id()]
}).

%%% ### Agent Record
%%%
%%% Agent - complete neural network with its genotype and training history
%%%
%%% **Fields:**
%%% - `id` - Unique agent identifier
%%% - `generation` - Current generation number
%%% - `population_id` - ID of the population this agent belongs to
%%% - `specie_id` - ID of the species this agent belongs to
%%% - `cortex_id` - ID of this agent's cortex
%%% - `fingerprint` - Generalized structural signature for speciation
%%% - `constraint` - Morphology and neural constraints
%%% - `evolution_history` - List of mutations applied to this agent
%%% - `fitness` - Latest fitness score (higher is better)
%%% - `innovation_factor` - Innovation metric for novelty
%%% - `pattern` - Network topology pattern [{LayerIndex, NeuronCount}]
-record(agent, {
    id :: agent_id(),
    generation :: non_neg_integer(),
    population_id :: population_id(),
    specie_id :: specie_id(),
    cortex_id :: cortex_id(),
    fingerprint :: term(),
    constraint :: #constraint{},
    evolution_history = [] :: [tuple()],
    fitness :: float() | undefined,
    innovation_factor = 0 :: non_neg_integer(),
    pattern = [] :: [{layer_index(), pos_integer()}]
}).

%%% ### Specie Record
%%%
%%% Species - group of agents with similar structural fingerprints
%%%
%%% **Fields:**
%%% - `id` - Unique species identifier
%%% - `population_id` - ID of the population containing this species
%%% - `fingerprint` - Structural fingerprint shared by all members
%%% - `constraint` - Morphology constraints for this species
%%% - `agent_ids` - List of agent IDs in this species
%%% - `dead_pool` - List of eliminated agent IDs
%%% - `champion_ids` - List of top-performing agent IDs
%%% - `fitness` - Species average fitness
%%% - `innovation_factor` - Species innovation metric
-record(specie, {
    id :: specie_id(),
    population_id :: population_id(),
    fingerprint :: term(),
    constraint :: #constraint{},
    agent_ids = [] :: [agent_id()],
    dead_pool = [] :: [agent_id()],
    champion_ids = [] :: [agent_id()],
    fitness :: float() | undefined,
    innovation_factor = 0 :: non_neg_integer()
}).

%%% ### Population Record
%%%
%%% Population - collection of species evolving together
%%%
%%% **Fields:**
%%% - `id` - Unique population identifier
%%% - `polis_id` - ID of the platform managing this population
%%% - `specie_ids` - List of species IDs in this population
%%% - `morphologies` - List of allowed morphologies
%%% - `innovation_factor` - Population-wide innovation metric
-record(population, {
    id :: population_id(),
    polis_id :: atom(),
    specie_ids = [] :: [specie_id()],
    morphologies = [] :: [morphology()],
    innovation_factor :: non_neg_integer()
}).
