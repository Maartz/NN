-module(neuron).
-compile(export_all).
-include("records.hrl").
-define(DELTA_MULTIPLIER,math:pi()*2).
-define(SAT_LIMIT,math:pi()*2).

gen(ExoSelf_PId,Node)->
	spawn(Node,?MODULE,prep,[ExoSelf_PId]).

prep(ExoSelf_PId) ->
    rand:seed(exsss),
	receive 
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
			loop(Id,ExoSelf_PId,Cx_PId,AF,{[{Input_PId,Weights}|Input_PIdPs],MInput_PIdPs},Output_PIds,Acc);
		{ExoSelf_PId,weight_restore}->
			RInput_PIdPs = get(weights),
			loop(Id,ExoSelf_PId,Cx_PId,AF,{RInput_PIdPs,RInput_PIdPs},Output_PIds,Acc);
		{ExoSelf_PId,weight_perturb}->
			PInput_PIdPs=perturb_IPIdPs(MInput_PIdPs),
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