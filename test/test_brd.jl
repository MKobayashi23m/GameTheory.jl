# ------------------------------- #
# Testing best response dynamics  #
# ------------------------------- #

using Random

@testset "Testing brd.jl" begin
    
    payoff_matrix = [4 0; 3 2]
    N = 4
    ts_length = 3
    init_action_dist = [4, 0]

    @testset "Testing best response dynamics model" begin
        
        brd = BRD(payoff_matrix, N)
        @test @inferred(play(brd, init_action_dist, num_reps=ts_length)) ==
              [4, 0]
        @test @inferred(time_series(brd, ts_length, init_action_dist)) ==
              [4 4 4; 0 0 0]
    end

    @testset "Testing KMR model" begin
        
        epsilon = 0.1
        seed = 1234
        kmr = KMR(payoff_matrix, N, epsilon)
        @test @inferred(play(MersenneTwister(seed), kmr, init_action_dist,
                             num_reps=ts_length)) ==
                play(MersenneTwister(seed), kmr, init_action_dist,
                             num_reps=ts_length)
        @test @inferred(time_series(MersenneTwister(seed), kmr, ts_length,
                                    init_action_dist)) ==
                time_series(MersenneTwister(seed), kmr, ts_length,
                                    init_action_dist)
    end

    @testset "Testing sampling best response dynamics model" begin
        
        k = 2
        seed = 1234
        sbrd = SamplingBRD(payoff_matrix, N, k)
        @test @inferred(play(MersenneTwister(seed), sbrd, init_action_dist,
                             num_reps=ts_length)) ==
                play(MersenneTwister(seed), sbrd, init_action_dist,
                             num_reps=ts_length)
        @test @inferred(time_series(MersenneTwister(seed), sbrd, ts_length,
                                    init_action_dist)) ==
                time_series(MersenneTwister(seed), sbrd, ts_length,
                                    init_action_dist)
    end
end