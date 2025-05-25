-module(actuator).
-compile(export_all).
-include("records.hrl").

gen(ExoSelf_PId,Node)->
	spawn(Node,?MODULE,prep,[ExoSelf_PId]).

prep(ExoSelf_PId) -> 
	receive 
		{ExoSelf_PId,{Id,Cx_PId,Scape,ActuatorName,Fanin_PIds}} ->
			loop(Id,ExoSelf_PId,Cx_PId,Scape,ActuatorName,{Fanin_PIds,Fanin_PIds},[])
	end.

loop(Id,ExoSelf_PId,Cx_PId,Scape,AName,{[From_PId|Fanin_PIds],MFanin_PIds},Acc) ->
	receive
		{From_PId,forward,Input} ->
			loop(Id,ExoSelf_PId,Cx_PId,Scape,AName,{Fanin_PIds,MFanin_PIds},lists:append(Input,Acc));
		{ExoSelf_PId,terminate} ->
			ok
	end;
loop(Id,ExoSelf_PId,Cx_PId,Scape,AName,{[],MFanin_PIds},Acc)->
    {Fitness,EndFlag} = actuator:AName(lists:reverse(Acc),Scape),
    io:format("Actuator ~p: Got fitness ~p, endflag ~p~n", [Id, Fitness, EndFlag]),
    Cx_PId ! {self(),sync,Fitness,EndFlag},
    loop(Id,ExoSelf_PId,Cx_PId,Scape,AName,{MFanin_PIds,MFanin_PIds},[]).


pts(Result,_Scape)->
	io:format("actuator:pts(Result): ~p~n",[Result]),
	{1,0}.

xor_SendOutput(Output,Scape)->
	Scape ! {self(),action,Output},
	receive 
		{Scape,Fitness,HaltFlag}->
			{Fitness,HaltFlag}
	end.
