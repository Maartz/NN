# Feed-Forward Neural Network in Erlang/OTP

## Overview
This project implements a Feed-Forward Neural Network (FFNN) using Erlang/OTP's actor model. Each component of the neural network (neurons, sensors, actuators, and cortex) is implemented as a separate process, allowing for concurrent execution and message passing.

## Architecture

The system consists of several key modules:

1. `constructor.erl` - Generates the network structure
2. `sensor.erl` - Handles input generation
3. `neuron.erl` - Implements neuron behavior
4. `actuator.erl` - Manages output processing
5. `cortex.erl` - Orchestrates the network operation
6. `records.hrl` - Defines data structures

### Key Components

```erlang
-record(sensor, {id, cortex_id, name, vector_length, fanout_ids}).
-record(actuator, {id, cortex_id, name, vector_length, fanin_ids}).
-record(neuron, {id, cortex_id, activation_function, input_ids, output_ids}).
-record(cortex, {id, sensor_ids, actuator_ids, neuron_ids}).
```

## Setup

1. Ensure Erlang/OTP 26 or later is installed
2. Compile all modules:
```erlang
c(constructor).
c(sensor).
c(neuron).
c(actuator).
c(cortex).
```

## Usage

To create a new neural network:

```erlang
constructor:construct_genotype("ffnn.erl", rng, pts, [1,3]).
```

Parameters:
- `"ffnn.erl"` - Output file name
- `rng` - Sensor type (random number generator)
- `pts` - Actuator type (prints to screen)
- `[1,3]` - Hidden layer configuration (1 neuron in first hidden layer, 3 in second)

## Network Flow

1. **Initialization**: The constructor generates network topology
2. **Sensor**: Generates input using the RNG sensor
3. **Neural Layers**: Process data through configured layers
4. **Actuator**: Outputs results through the PTS (print to screen) actuator

## Resources

For learning more about Erlang/OTP and neural networks:

- [Learn You Some Erlang](https://learnyousomeerlang.com/) - Excellent Erlang tutorial
- [Erlang Documentation](https://www.erlang.org/docs) - Official documentation
- [Making reliable distributed systems in the presence of software errors](https://erlang.org/download/armstrong_thesis_2003.pdf) - Joe Armstrong's thesis on Erlang

## Example Session

Here's a typical session:

```erlang
Eshell V14.1.1
1> c(constructor).
{ok,constructor}
2> constructor:construct_genotype("ffnn.erl",rng,pts,[1,3]).
ok
```

This will:
1. Generate a neural network configuration
2. Save it to `ffnn.erl`
3. Create a network with:
   - Random number generator input
   - Two hidden layers (1 and 3 neurons)
   - Print-to-screen output

## Implementation Notes

- Uses Erlang's actor model for concurrent processing
- Each neuron runs as a separate process
- Communication happens via message passing
- Uses hyperbolic tangent (tanh) as activation function
- Supports dynamic network topology

Let me know if you need any clarification or have questions about specific parts of the implementation!