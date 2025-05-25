-module(test_nn).
-compile(export_all).

test() ->
    io:format("Creating genotype...~n"),
    genotype:construct(test_nn_genotype, xor_mimic, [2]),
    
    io:format("Starting network...~n"),
    Pid = exoself:map(test_nn_genotype),
    
    Ref = monitor(process, Pid),
    
    receive
        {'DOWN', Ref, process, Pid, Reason} ->
            io:format("ExoSelf terminated with reason: ~p~n", [Reason])
    after 10000 ->
        io:format("Network still running after 10 seconds~n")
    end.
