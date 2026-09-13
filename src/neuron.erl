-module(neuron).
-compile(export_all).
-include("records.hrl").
%% ATELIER : la variation vaut (uniform()-0.5)*DELTA_MULTIPLIER, donc environ
%% [-pi, +pi] avec ce réglage. SAT_LIMIT borne le poids final à +/-2*pi.
%% Ce sont deux paramètres distincts : amplitude du saut et borne des poids.
%% Les garder fixes pour comparer uniquement les redémarrages (docs/TRAIN.md).
-define(DELTA_MULTIPLIER,math:pi()*2).
-define(SAT_LIMIT,math:pi()*2).

gen(ExoSelf_PId,Node)->
	case rpc:call(Node, erlang, whereis, [platform]) of
        undefined -> spawn(Node, ?MODULE, prep, [ExoSelf_PId]);
        Pid when is_pid(Pid) ->
            gen_server:call({platform, Node}, {spawn_neuron, ExoSelf_PId})
    end.

prep(ExoSelf_PId) ->
    link(ExoSelf_PId),
    rand:seed(exsplus),
	receive 
		{ExoSelf_PId, seed, Seed} ->
            rand:seed(exsplus, Seed),
            receive
                {ExoSelf_PId,{Id,Cx_PId,AF,Input_PIdPs,Output_PIds}} ->
                    loop(Id,ExoSelf_PId,Cx_PId,AF,{Input_PIdPs,Input_PIdPs},Output_PIds,0)
            end;
        {ExoSelf_PId,{Id,Cx_PId,AF,Input_PIdPs,Output_PIds}} ->
			loop(Id,ExoSelf_PId,Cx_PId,AF,{Input_PIdPs,Input_PIdPs},Output_PIds,0)
	end.

loop(Id,ExoSelf_PId,Cx_PId,AF,{[{Input_PId,Weights}|Input_PIdPs],MInput_PIdPs},Output_PIds,Acc)->
	receive
		{Input_PId,forward,Input}->
			Result = dot(Input,Weights,0),
			loop(Id,ExoSelf_PId,Cx_PId,AF,{Input_PIdPs,MInput_PIdPs},Output_PIds,Result+Acc);
		{ExoSelf_PId,weight_backup}->
			put(weights,MInput_PIdPs),
            ExoSelf_PId ! {self(), weights_updated},
			loop(Id,ExoSelf_PId,Cx_PId,AF,{[{Input_PId,Weights}|Input_PIdPs],MInput_PIdPs},Output_PIds,Acc);
		{ExoSelf_PId,weight_restore}->
			RInput_PIdPs = get(weights),
            ExoSelf_PId ! {self(), weights_updated},
			loop(Id,ExoSelf_PId,Cx_PId,AF,{RInput_PIdPs,RInput_PIdPs},Output_PIds,Acc);
		{ExoSelf_PId,weight_perturb}->
			PInput_PIdPs=perturb_IPIdPs(MInput_PIdPs),
            ExoSelf_PId ! {self(), weights_updated},
			loop(Id,ExoSelf_PId,Cx_PId,AF,{PInput_PIdPs,PInput_PIdPs},Output_PIds,Acc);
		{ExoSelf_PId,get_backup}->
			ExoSelf_PId ! {self(),Id,MInput_PIdPs},
			loop(Id,ExoSelf_PId,Cx_PId,AF,{[{Input_PId,Weights}|Input_PIdPs],MInput_PIdPs},Output_PIds,Acc);
		{ExoSelf_PId,terminate}->
			ok
	end;
loop(Id,ExoSelf_PId,Cx_PId,AF,{[Bias],MInput_PIdPs},Output_PIds,Acc)->
	Output = neuron:AF(Acc+Bias),
	[Output_PId ! {self(),forward,[Output]} || Output_PId <- Output_PIds],
	loop(Id,ExoSelf_PId,Cx_PId,AF,{MInput_PIdPs,MInput_PIdPs},Output_PIds,0);
loop(Id,ExoSelf_PId,Cx_PId,AF,{[],MInput_PIdPs},Output_PIds,Acc)->
	Output = neuron:AF(Acc),
	[Output_PId ! {self(),forward,[Output]} || Output_PId <- Output_PIds],
	loop(Id,ExoSelf_PId,Cx_PId,AF,{MInput_PIdPs,MInput_PIdPs},Output_PIds,0).
	
dot([I|Input],[W|Weights],Acc) ->
	dot(Input,Weights,I*W+Acc);
dot([],[],Acc)->
	Acc.

tanh(Val)->
	math:tanh(Val).

perturb_IPIdPs(Input_PIdPs)->
    %% Deuxième tirage, après la sélection des neurones par ExoSelf :
    %% chaque coefficient a une probabilité 1/sqrt(Tot_Weights) de changer.
    %% Tot_Weights compte les connexions ; le biais utilise la même probabilité.
    %% Ce n'est pas un gradient : la direction du changement est aléatoire.
	Tot_Weights=lists:sum([length(Weights) || {_Input_PId,Weights}<-Input_PIdPs]),
	MP = 1/math:sqrt(Tot_Weights),
	perturb_IPIdPs(MP,Input_PIdPs,[]).
perturb_IPIdPs(MP,[{Input_PId,Weights}|Input_PIdPs],Acc)->
	U_Weights = perturb_weights(MP,Weights,[]),
	perturb_IPIdPs(MP,Input_PIdPs,[{Input_PId,U_Weights}|Acc]);
perturb_IPIdPs(MP,[Bias],Acc)->
	U_Bias = case rand:uniform() < MP of
		true-> sat((rand:uniform()-0.5)*?DELTA_MULTIPLIER+Bias,-?SAT_LIMIT,?SAT_LIMIT);
		false -> Bias
	end,
	lists:reverse([U_Bias|Acc]);
perturb_IPIdPs(_MP,[],Acc)->
	lists:reverse(Acc).

perturb_weights(MP,[W|Weights],Acc)->
    %% ATELIER / "sans effet" : comparer W et U_W APRÈS sat/3.
    %% Un tirage peut demander un changement mais la borne peut l'annuler.
    %% L'ACK weights_updated confirme la fin de la commande, pas une mutation
    %% effective. Ne pas le compter comme un poids changé sans comparaison.
    U_W = case rand:uniform() < MP of
        true->
            sat((rand:uniform()-0.5)*?DELTA_MULTIPLIER+W,-?SAT_LIMIT,?SAT_LIMIT);
        false ->
            W
    end,
    perturb_weights(MP,Weights,[U_W|Acc]);
perturb_weights(_MP,[],Acc)->
    lists:reverse(Acc).

sat(Val,Min,Max)->
  if
      Val < Min -> Min;
      Val > Max -> Max;
      true -> Val
  end.
