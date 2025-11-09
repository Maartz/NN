-record(sensor, {id, cortex_id, name, scape, vector_length, fanout_ids}).
-record(actuator, {id, cortex_id, name, scape, vector_length, fanin_ids}).
-record(neuron, {id, cortex_id, activation_function, input_ids, output_ids}).
-record(cortex, {id, sensor_ids, actuator_ids, neuron_ids}).
-record(agent, {id, generation, population_id, specie_id, cortex_id, fingerprint, constraint, evolution_history=[], fitness, innovation_factor=0, pattern=[]}).
-record(specie, {id, population_id, fingerprint, constraint, agent_ids=[], dead_pool=[], champion_ids=[], fitness, innovation_factor=0}).
-record(population, {id, platform_id, specie_ids=[], morphologies=[], innovation_factor}).
-record(constraint, {
    morphology=xor_mimic,
    neural_afs=[tanh, cos, gauss, abs]
         }).
