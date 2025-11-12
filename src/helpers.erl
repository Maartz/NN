%%% # Helper Functions for Neural Network ID Generation
%%%
%%% This module provides utility functions for generating unique identifiers
%%% used throughout the neural network system.
%%%
%%% ## ID Generation Strategy
%%%
%%% IDs are generated as floating-point numbers using Erlang's monotonic
%%% unique integer generator. The formula is:
%%%
%%% ```
%%% ID = 1.0 / (UniqueInt + 1)
%%% ```
%%%
%%% **Properties:**
%%% - Unique within the same Erlang runtime instance
%%% - Monotonically decreasing (later IDs are smaller)
%%% - Always positive and ≤ 1.0
%%% - **NOT cryptographically secure** - predictable and forgeable
%%%
%%% **Security Note:** These IDs are suitable for internal component
%%% identification but NOT for security-sensitive applications. For secure
%%% random IDs, use `crypto:strong_rand_bytes/1`.

-module(helpers).
-compile(export_all).

-export([generate_id/0, generate_ids/2]).

%% @doc Generate a single unique ID
%%
%% Generates a unique floating-point identifier using Erlang's monotonic
%% unique integer generator.
%%
%% === Examples ===
%% ```
%% helpers:generate_id().
%% 0.5
%% helpers:generate_id().
%% 0.3333333333333333
%% '''
-spec generate_id() -> float().
generate_id() ->
    UniqueInt = erlang:unique_integer([positive, monotonic]),
    1.0 / (UniqueInt + 1).

%% @doc Generate N unique IDs
%%
%% Generates a list of N unique floating-point identifiers. Each ID is
%% guaranteed to be unique within the same Erlang runtime instance.
%%
%% The IDs are generated using `generate_id/0' and accumulated in a list.
%%
%% === Examples ===
%% ```
%% helpers:generate_ids(3, []).
%% [0.5, 0.3333333333333333, 0.25]
%%
%% helpers:generate_ids(0, []).
%% []
%% '''
-spec generate_ids(Count :: non_neg_integer(), Acc :: [float()]) -> [float()].
generate_ids(0, Acc) ->
    Acc;
generate_ids(Index, Acc) ->
    Id = generate_id(),
    generate_ids(Index - 1, [Id | Acc]).
