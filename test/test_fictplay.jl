# ------------------------- #
# Testing fictitious play   #
# ------------------------- #


using Distributions
using Random


@testset "Testing fictplay.jl" begin

    matching_pennies_bimatrix = Array{Float64}(undef, 2, 2, 2)
    matching_pennies_bimatrix[:, 1, 1] = [1, -1]
    matching_pennies_bimatrix[:, 1, 2] = [-1, 1]
    matching_pennies_bimatrix[:, 2, 1] = [-1, 1]
    matching_pennies_bimatrix[:, 2, 2] = [1, -1]
    g = NormalFormGame(matching_pennies_bimatrix)

    gain = 0.1
    init_actions = (1,1)

    function vector_approximate_equal(vec1::NTuple{2,Vector{T1}},
                                      vec2::NTuple{2,Vector{T2}}) where {T1,T2}
        @test T1 == T2
        for (x1, x2) in zip(vec1, vec2)
            @test length(x1) == length(x2)
            for (xx1, xx2) in zip(x1, x2)
                @test xx1 ≈ xx2
            end
        end
    end

    function matrix_approximate_equal(mat1::NTuple{2,Matrix{T1}},
                                      mat2::NTuple{2,Matrix{T2}}) where {T1,T2}
        @test T1 == T2
        for (x1, x2) in zip(mat1, mat2)
            @test size(x1) == size(x2)
            row, col = size(x1)
            for i in 1:row
                for j in 1:col
                    @test x1[i, j] ≈ x2[i, j]
                end
            end
        end
    end

    @testset "Testing fictitious play" begin

        fp_dec = FictitiousPlay(g)
        x = play(fp_dec, init_actions)
        x_series = time_series(fp_dec, 3, init_actions)
        x_des = ([1.0, 0.0], [0.5, 0.5])
        x_series_des = ([1.0 1.0 1.0; 0.0 0.0 0.0], [1.0 0.5 1/3; 0.0 0.5 2/3])

        fp_con = FictitiousPlay(g, ConstantGain(gain))
        y = play(fp_con, init_actions)
        y_series = time_series(fp_con, 3, init_actions)
        y_des = ([1.0, 0.0], [0.9, 0.1])
        y_series_des = (
                        [1.0 1.0 1.0; 0.0 0.0 0.0], [1.0 0.9 0.81; 0.0 0.1 0.19]
                       )

        vector_approximate_equal(x, x_des)
        vector_approximate_equal(y, y_des)
        matrix_approximate_equal(x_series, x_series_des)
        matrix_approximate_equal(y_series, y_series_des)
    end

    @testset "Testing stochastic fictitious play" begin

        normal = Normal()  #standard normal distribution
        seed = 1234  #seed for random number generator

        sfp_dec = StochasticFictitiousPlay(g, normal)
        x = [play(MersenneTwister(seed), sfp_dec, init_actions) for i in 1:2]
        x_series = [time_series(MersenneTwister(seed), sfp_dec, 3, init_actions)
                    for i in 1:2]

        sfp_con = StochasticFictitiousPlay(g, normal, ConstantGain(gain))
        y = [play(MersenneTwister(seed), sfp_con, init_actions) for i in 1:2]
        y_series = [time_series(MersenneTwister(seed), sfp_con, 3, init_actions)
                    for i in 1:2]

        vector_approximate_equal(x[1], x[2])
        vector_approximate_equal(y[1], y[2])
        matrix_approximate_equal(x_series[1], x_series[2])
        matrix_approximate_equal(y_series[1], y_series[2])
    end
end