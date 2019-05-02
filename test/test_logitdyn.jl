# ------------------------------------- #
# Testing Logit-Dynamics Response model #
# ------------------------------------- #

using Random

@testset "Testing logitdyn.jl" begin

    payoff_matrix = [4 0; 3 2]
    beta = 4.0
    g = NormalFormGame(payoff_matrix)
    lgdy = LogitDynamics(g, beta)

    seed = 1234
    init_actions = (1, 1)
    ts_length = 3
    @test play(MersenneTwister(seed), lgdy, init_actions) ==
            play(MersenneTwister(seed), lgdy, init_actions)
    @test time_series(MersenneTwister(seed), lgdy, ts_length, init_actions) ==
            time_series(MersenneTwister(seed), lgdy, ts_length, init_actions)
end 
