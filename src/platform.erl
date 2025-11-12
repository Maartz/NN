-module(platform).

-export([start/1,start/0,stop/0,init/2,create/0,reset/0,sync/0]).
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,terminate/2, code_change/3]).

-behaviour(gen_server).
-include("records.hrl").

-record(?MODULE, {active_mods=[], active_scapes=[]}).
-record(scape_summary, {address, type, parameters=[]}).
-define(MODS, []).
-define(PUBLIC_SCAPES, []).

sync() ->
  make:all([load]).

start() ->
  case whereis(platform) of
    undefined ->
      gen_server:start(?MODULE, {?MODS, ?PUBLIC_SCAPES}, []);
    PlatformPId ->
      io:format("Platform: ~p is already running on this node. ~n", [PlatformPId])
  end.

start(StartParameters) ->
  gen_server:start(?MODULE, StartParameters, []).
init(PId, InitState) ->
  gen_server:cast(PId, {init, InitState}).

stop() ->
  case whereis(platform) of
    undefined ->
      io:format("Platform cannot be stopped, it is not online~n");
    PlatformPId ->
      gen_server:cast(PlatformPId, {stop, normal})
  end.

init({Mods, PublicScapes}) ->
  rand:seed(exsplus, os:timestamp()),
  process_flag(trap_exit, true),
  register(platform, self()),
  io:format("Parameters: ~p~n", [{Mods, PublicScapes}]),
  mnesia:start(),
  start_supmods(Mods),
  ActivePublicScapes = start_scapes(PublicScapes, []),
  io:format("###*** Platform is now online ***###~n"),
  InitState = #?MODULE{active_mods=Mods, active_scapes=ActivePublicScapes},
  {ok, InitState}.

handle_call({get_scape, Type}, {CortexPId, _Ref}, State) ->
  ActivePublicScapes = State#?MODULE.active_scapes,
  ScapePId = case lists:keyfind(Type, 3, ActivePublicScapes) of
              false -> undefined;
              PS ->
                 PS#scape_summary.address
             end,
  {reply, ScapePId, State};
handle_call({stop, normal}, _From, State) ->
  {stop, normal, State};
handle_call({stop, shutdown}, _From, State) ->
  {stop, shutdown, State}.

handle_cast({init, InitState}, State) ->
  {noreply, State};
handle_cast({stop, normal}, State) ->
  {stop, normal, State};
handle_cast({stop, shutdown}, State) ->
  {stop, shutdown, State}.

handle_info(_Info, State) ->
  {noreply, State}.

terminate(Reason, S) ->
  ActiveMods = S#?MODULE.active_mods,
  stop_supmods(ActiveMods),
  stop_scapes(S#?MODULE.active_scapes),
  io:format("###*** Platform is now offline ***###~nTerminated with reason: ~p~n", [Reason]),
  ok.

code_change(_OldVersion, State, _Extra) ->
  {ok, State}.

create() ->
  mnesia:create_schema([node()]),
  mnesia:start(),
  mnesia:create_table(population, [{disc_copies, [node()]}, {type, set}, {attributes, record_info(fields, population)}]),
  mnesia:create_table(specie, [{disc_copies, [node()]}, {type, set}, {attributes, record_info(fields, specie)}]),
  mnesia:create_table(agent, [{disc_copies, [node()]}, {type, set}, {attributes, record_info(fields, agent)}]),
  mnesia:create_table(cortex, [{disc_copies, [node()]}, {type, set}, {attributes, record_info(fields, cortex)}]),
  mnesia:create_table(neuron, [{disc_copies, [node()]}, {type, set}, {attributes, record_info(fields, neuron)}]),
  mnesia:create_table(sensor, [{disc_copies, [node()]}, {type, set}, {attributes, record_info(fields, sensor)}]),
  mnesia:create_table(actuator, [{disc_copies, [node()]}, {type, set}, {attributes, record_info(fields, actuator)}]).

reset() ->
  mnesia:stop(),
  ok = mnesia:delete_schema([node()]),
  platform:create().

start_supmods([ModName | ActiveMods]) ->
  ModName:start(),
  stop_supmods(ActiveMods);
start_supmods([]) ->
  done.

stop_supmods([ModName | ActiveMods]) ->
  ModName:stop(),
  stop_supmods(ActiveMods);
stop_supmods([]) ->
  done.

start_scapes([S | Scapes], Acc) ->
  Type = S#scape_summary.type,
  Parameters = S#scape_summary.parameters,
  {ok, Pid} = scape:start_link({self(), Type, Parameters}),
  start_scapes(Scapes, [S#scape_summary{address=Pid} | Acc]);
start_scapes([], Acc) ->
  lists:reverse(Acc).

stop_scapes([S | Scapes]) ->
  Pid = S#scape_summary.address,
  gen_server:cast(Pid, {self(), stop, normal}),
  stop_scapes(Scapes);
stop_scapes([]) ->
  ok.
