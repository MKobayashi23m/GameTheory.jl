#=
Tools for repeated games

This file contains code to build and manage repeated games

It currently only has tools for solving two player repeated
games, but could be extended to do more with some effort.

March, 2020
Updated to adjust exact arithmetic
For the implementation, make sure all of your imputs are `Rational` or `Int`.
Also, use packages like `CDDlib` for calculations.
=#

"""
    RepeatedGame{N,T,TD}

Type representing an N-player repeated game.

# Fields

- `sg::NormalFormGame{N, T}` : The stage game used to create the repeated game.
- `delta::TD` : The common discount rate at which all players discount the
  future.
"""
struct RepeatedGame{N, T<:Real, TD<:Real}
    sg::NormalFormGame{N, T}
    delta::TD
end

# Type alias for 2 player game
"""
    RepGame2

Type representing a 2-player repeated game; alias for `RepeatedGame{2}`.
"""
const RepGame2 = RepeatedGame{2}

#
# Helper Functions (for 2 players)
#

"""
    RepeatedGame(p1, p2, delta)

Helper constructor that builds a repeated game for two players.

# Arguments

- `p1::Player` : The first player.
- `p2::Player` : The second player.
- `delta::TD` : The common discount rate at which all players discount the
  future.

# Returns

- `::RepeatedGame` : The repeated game.
"""
RepeatedGame(p1::Player, p2::Player, delta::TD) where TD<:Type =
    RepeatedGame(NormalFormGame((p1, p2)), delta)

"""
    unpack(rpd)

Helper function that unpacks the elements of a repeated game.

# Arguments

- `rpd::RepeatedGame` : The repeated game.

# Returns

- `::Tuple{NormalFormGame, TD<:Real}` : A tuple containing the stage game and
the delta.
"""
unpack(rpd::RepeatedGame) = (rpd.sg, rpd.delta)

# Flow utility in terms of the players actions
flow_u_1(rpd::RepGame2, a1::Int, a2::Int) =
    rpd.sg.players[1].payoff_array[a1, a2]
flow_u_2(rpd::RepGame2, a1::Int, a2::Int) =
    rpd.sg.players[2].payoff_array[a2, a1]
flow_u(rpd::RepGame2, a1::Int, a2::Int) =
    [flow_u_1(rpd, a1, a2), flow_u_2(rpd, a1, a2)]

# Computes each players best deviation given an opponent's action
best_dev_i(rpd::RepGame2, i::Int, aj::Int) =
    argmax(rpd.sg.players[i].payoff_array[:, aj])
best_dev_1(rpd::RepGame2, a2::Int) = best_dev_i(rpd, 1, a2)
best_dev_2(rpd::RepGame2, a1::Int) = best_dev_i(rpd, 2, a1)

# Computes the payoff of the best deviation
best_dev_payoff_i(rpd::RepGame2, i::Int, aj::Int) =
    maximum(rpd.sg.players[i].payoff_array[:, aj])
best_dev_payoff_1(rpd::RepGame2, a2::Int) =
    maximum(rpd.sg.players[1].payoff_array[:, a2])
best_dev_payoff_2(rpd::RepGame2, a1::Int) =
    maximum(rpd.sg.players[2].payoff_array[:, a1])

"""
    sqpts(npts,TD)

Places `npts` equally spaced points along the 2 dimensional square with
vertices (1,0), (0,1), (-1,0) and (0,-1). This function returns the points
with x coordinates in first column and y coordinates in second column.

# Arguments

- `npts::Int` : Number of points to be placed.
- `TD<:Real` : Type of the discount factor in the repeated game.
# Returns

- `pts::Matrix{TD}` : Matrix of shape `(nH, 2)` containing the coordinates
  of the points.
"""
function sqpts(npts::Int, TD::TP) where TP <:Type
    # Want our points placed on [0, 1]
    incr = convert(TD,1 // npts)
    degrees = zero(TD):incr:one(TD)

    # Points on the first quadrant
    pts = Array{TD}(undef, 4npts, 2)
    for i=1:npts
        x = degrees[i]
        pts[i, 1] = x
        pts[i, 2] = 1 - x
    end
    # Points on the second quadrant
    for i = npts+1:2npts
        pts[i,1] = -pts[i-npts,1]
        pts[i,2] = pts[i-npts,2]
    end
    # The third quadrant
    for i = 2npts+1:3npts
        pts[i,1] = -pts[i-2npts,1]
        pts[i,2] = -pts[i-2npts,2]
    end
    # The fourth quadrants
    for i = 3npts+1:4npts
        pts[i,1] = pts[i-3npts,1]
        pts[i,2] = -pts[i-3npts,2]
    end
    return pts
end

"""
    initialize_sg_hpl(TD, nH, o, r)

Initializes subgradients, extreme points and hyperplane levels for the
approximation of the convex value set of a 2 player repeated game.

# Arguments

- `TD::TP` : Type of the discount factor
- `nH::Int` : Number of subgradients used for the approximation.
- `o::Vector{<Real}` : Origin for the approximation.
- `r::TR` : Radius for the approximation.


# Returns

- `C::Vector{TD}` : Vector of length `nH` containing the hyperplane levels.
- `H::Matrix{TD}` : Matrix of shape `(nH, 2)` containing the subgradients.
- `Z::Matrix{TD}` : Matrix of shape `(nH, 2)` containing the extreme points of
  the value set.
"""
function initialize_sg_hpl(TD::TP where TP<:Type, nH::Int, o::Vector{<:Real},
    r::TR where TR<:Real)
    # First create points on the square
    H = sqpts(nH,TD)
    HT = H'

    # Choose origin and radius for big approximation
    Z = Array{TD}(undef, 2, nH)
    for i=1:nH
        # We know that players can ever get worse than their
        # lowest punishment, so ignore anything below that
        Z[1, i] = o[1] + r*HT[1, i]
        Z[2, i] = o[2] + r*HT[2, i]
    end

    # Corresponding hyperplane levels
    C = dropdims(sum(HT .* Z, dims=1), dims=1)

    return C, H, Z
end

"""
    initialize_sg_hpl(rpd, nH, TD)

Initializes subgradients, extreme points and hyperplane levels for the
approximation of the convex value set of a 2 player repeated game by choosing
an appropriate origin and radius.

# Arguments

- `rpd::RepeatedGame` : Two player repeated game.
- `nH::Int` : Number of subgradients used for the approximation.
- `TD::TP` : Type of the discount factor
# Returns

- `C::Vector{TD}` : Vector of length `nH` containing the hyperplane levels.
- `H::Matrix{TD}` : Matrix of shape `(nH, 2)` containing the subgradients.
- `Z::Matrix{TD}` : Matrix of shape `(nH, 2)` containing the extreme points of
  the value set.
"""
function initialize_sg_hpl(rpd::RepeatedGame, nH::Int, TD::TP where TP<:Type)
    # Choose the origin to be mean of max and min payoffs
    p1_min, p1_max = extrema(rpd.sg.players[1].payoff_array)
    p2_min, p2_max = extrema(rpd.sg.players[2].payoff_array)

    o = [convert(TD,(p1_min + p1_max)//2), convert(TD,(p2_min + p2_max)//2)]
    r1 = max((p1_max - o[1])^2, (o[1] - p1_min)^2)
    r2 = max((p2_max - o[2])^2, (o[2] - p2_min)^2)
    r = convert(TD, sqrt(r1 + r2))

    return initialize_sg_hpl(nH, o, r,TD)
end

#
# Linear Programming Functions
#
"""
    initialize_LP_matrices(rpd, TD, H)

Initialize matrices for the linear programming problems.

# Arguments

- `rpd::RepeatedGame` : Two player repeated game.
- `TD::TP` : Type of the discount factor in the repeated game.
- `H::Matrix{<:Real}` : Matrix of shape `(nH, 2)` containing the subgradients
  used to approximate the value set, where `nH` is the number of subgradients.

# Returns

- `c::Vector{TD}` : Vector of length `nH` used to determine which
  subgradient should be used, where `nH` is the number of subgradients.
- `A::Matrix{TD}` : Matrix of shape `(nH+2, 2)` with nH set
  constraints and to be filled with 2 additional incentive compatibility
  constraints.
- `b::Vector{TD}` : Vector of length `nH+2` to be filled with
  the values for the constraints.
"""
function initialize_LP_matrices(rpd::RepGame2, TD:: TP where TP<:Type,
    H::Matrix{<:Real})
    # Need total number of subgradients
    nH = size(H, 1)

    # Create the c vector (objective)
    c = zeros(TD, 2)

    # Create the A matrix (constraints)
    A_H = H
    A_IC_1 = zeros(TD, 1, 2)
    A_IC_2 = zeros(TD, 1, 2)
    A_IC_1[1, 1] = -rpd.delta
    A_IC_2[1, 2] = -rpd.delta
    A = vcat(A_H, A_IC_1, A_IC_2)

    # Create the b vector (constraints)
    b = Array{TD}(undef, nH + 2)

    return c, A, b
end

"""
    worst_value_i(TD, rpd, H, C, i)

Given a constraint w ∈ W, this finds the worst possible payoff for agent i.

# Arguments

- `TD::TP` : Type of the discount factor in the repeated game.
- `rpd::RepGame2` : Two player repeated game.
- `H::Matrix{TM}` : Matrix of shape `(nH, 2)` containing the subgradients
  here `nH` is the number of subgradients.
- `C::Vector{TC}` : The array containing the hyperplane levels.
- `i::Int` : The player of interest.
- `lp_solver::Union{Type{<:MathOptInterface.AbstractOptimizer},Function}` :
  Linear programming solver to be used internally. Pass a
  `MathOptInterface.AbstractOptimizer` type (such as `Clp.Optimizer`) if no
  option is needed, or a function (such as `() -> Clp.Optimizer(LogLevel=0)`)
  to supply options.
  To implement exact arithmetic, use `CDDLib.Optimizer{TD}`, for example.


# Returns

- `out::TD` : Worst possible payoff for player i.
"""
function worst_value_i(TD::TP where TP<:Type,
    rpd::RepGame2, H::Matrix{TM} where TM<:Type,
    C::Vector{TC} where TC<:Type, i::Int,
    lp_solver::Union{Type{TO},Function}=() -> Clp.Optimizer(LogLevel=0)
) where {TO<:MOI.AbstractOptimizer}
    # Objective depends on which player we are minimizing
    c = zeros(TD,2)
    c[i] = one(TD)

    CACHE = MOIU.UniversalFallback(MOIU.Model{TD}())
    optimizer = MOIU.CachingOptimizer(CACHE, lp_solver())

    # Add variables
    x = MOI.add_variables(optimizer, 2)

    # Define objective function
    MOI.set(optimizer,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{TD}}(),
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), zero(TD)))
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Add constraints
    for i in 1:size(H,1)
        MOI.add_constraint(
            optimizer,
            MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(H[i, :],x), zero(TD)),
            MOI.LessThan(C[i])
        )
    end

    # Optimize
    MOI.optimize!(optimizer)

    status = MOI.get(optimizer, MOI.TerminationStatus())

    if status == MOI.OPTIMAL
        variable_result = MOI.get(optimizer, MOI.VariablePrimal(), x)
        out = variable_result[i]
    else
        out = minimum(rpd.sg.players[i].payoff_array)
    end

    return out
end

"See `worst_value_i` for documentation"
worst_value_1(
    TD::TP where TP<:Type,
    rpd::RepGame2,
    H::Matrix{TH} where TH<:Type,
    C::Vector{TC} where TC<:Type,
    lp_solver::Union{Type{TO},Function}=() -> Clp.Optimizer(LogLevel=0)
) where {TO<:MOI.AbstractOptimizer} = worst_value_i(rpd, H, C, 1, lp_solver)

"See `worst_value_i` for documentation"
worst_value_2(
    TD::TP where TP<:Type,
    rpd::RepGame2,
    H::Matrix{TH} where TH<:Type,
    C::Vector{TC} where TC<:Type,
    lp_solver::Union{Type{TO},Function}=() -> Clp.Optimizer(LogLevel=0)
) where {TO<:MOI.AbstractOptimizer} = worst_value_i(rpd, H, C, 2, lp_solver)

#
# Outer Hyper Plane Approximation
#
"""
    outerapproximation(TD,rpd; nH=32, tol=convert(TD,1e-8), maxiter=500, check_pure_nash=true,
                       verbose=false, nskipprint=50,
                       plib=default_library(2, Float64),
                       lp_solver=() -> Clp.Optimizer(LogLevel=0))

Approximates the set of equilibrium values for a repeated game with the outer
hyperplane approximation described by Judd, Yeltekin, Conklin (2002).

# Arguments

- `TD::TP` : Type of the discount factor.
- `rpd::RepGame2` : Two player repeated game.
- `nH::Int` : Number of subgradients used for the approximation.
- `tol::TT` : Tolerance in differences of set.
- `maxiter::Int` : Maximum number of iterations.
- `check_pure_nash`: Whether to perform a check about whether a pure Nash
  equilibrium exists.
- `verbose::Bool` : Whether to display updates about iterations and distance.
- `nskipprint::Int` : Number of iterations between printing information
  (assuming verbose=true).
- `plib::Polyhedra.Library`: Allows users to choose a particular package for
  the geometry computations.
  (See [Polyhedra.jl](https://github.com/JuliaPolyhedra/Polyhedra.jl)
  docs for more info). By default, it chooses to use `Polyhedra.DefaultLibrary`.
  To implement exact arithmetic, employ a package that support computations of
  rational numbers, such as `CDDlib`.
- `lp_solver::Union{<:Type{MathOptInterface.AbstractOptimizer},Function}` :
  Linear programming solver to be used internally. Pass a
  `MathOptInterface.AbstractOptimizer` type (such as `Clp.Optimizer`) if no
  option is needed, or a function (such as `() -> Clp.Optimizer(LogLevel=0)`)
  to supply options. For exact arithmetic, use `CDDLib.Optimizer{TD}`, for
  example.

# Returns

- `vertices::Matrix{TD}` : Vertices of the outer approximation of the
  value set.
"""
function outerapproximation(
        TD::TP where TP<:Type, rpd::RepGame2; nH::Int=32, tol::Float64=1e-8,
        maxiter::Int=500, check_pure_nash::Bool=true, verbose::Bool=false,
        nskipprint::Int=50, plib::Polyhedra.Library=default_library(2, Float64),
        lp_solver::Union{Type{TO},Function}=() -> Clp.Optimizer(LogLevel=0)
    ) where {TO<:MOI.AbstractOptimizer}

    # set up optimizer
    CACHE = MOIU.UniversalFallback(MOIU.Model{TD}())
    optimizer = MOIU.CachingOptimizer(CACHE, lp_solver())

    # Long unpacking of stuff
    sg, delta = unpack(rpd)
    p1, p2 = sg.players
    po_1, po_2 = p1.payoff_array, p2.payoff_array
    p1_minpayoff, p1_maxpayoff = extrema(po_1)
    p2_minpayoff, p2_maxpayoff = extrema(po_2)

    # Check to see whether at least one pure strategy NE exists
    pure_nash_exists = check_pure_nash ? length(pure_nash(sg; ntofind=1)) > 0 :
                       true
    if !pure_nash_exists
        error("No pure action Nash equilibrium exists in stage game")
    end

    # Get number of actions for each player and create action space
    nA1, nA2 = num_actions(p1), num_actions(p2)
    nAS = nA1 * nA2
    AS = QuantEcon.gridmake(1:nA1, 1:nA2)

    # Create the unit circle, points, and hyperplane levels
    C, H, Z = initialize_sg_hpl(rpd, nH, TD)
    Cnew = copy(C)

    # Create matrices for linear programming
    c, A, b = initialize_LP_matrices(rpd, TD, H)

    # Set iterative parameters and iterate until converged
    iter, dist = 0, 10
    while (iter < maxiter) & (dist > tol)
        # Compute the current worst values for each agent
        _w1 = worst_value_1(TD, rpd, H, C, lp_solver)
        _w2 = worst_value_2(TD, rpd, H, C, lp_solver)

        # Update all set constraints -- Copies elements 1:nH of C into b
        copyto!(b, 1, C, 1, nH)

        # Iterate over all subgradients
        for ih=1:nH
            #
            # Subgradient specific instructions
            #
            h1, h2 = H[ih, :]

            # Put the right objective into c (negative because want maximize)
            c[1] = -h1
            c[2] = -h2

            # Allocate space to store all solutions
            Cia = Array{TD}(undef, nAS)
            Wia = Array{TD}(undef, 2, nAS)
            for ia=1:nAS
                #
                # Action specific instruction
                #
                a1, a2 = AS[ia, :]

                # Update incentive constraints
                b[nH+1] = (1-delta)*flow_u_1(rpd, a1, a2) -
                          (1-delta)*best_dev_payoff_1(rpd, a2) - delta*_w1
                b[nH+2] = (1-delta)*flow_u_2(rpd, a1, a2) -
                          (1-delta)*best_dev_payoff_2(rpd, a1) - delta*_w2

                MOI.empty!(optimizer)

                # Add variables
                x = MOI.add_variables(optimizer, 2)

                # Define objective function
                MOI.set(
                    optimizer,
                    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{TD}}(),
                    MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(c, x), 0)
                    )
                MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

                # Add constraints
                for i in 1:size(A,1)
                    MOI.add_constraint(optimizer,
                                       MOI.ScalarAffineFunction(
                                       MOI.ScalarAffineTerm.(A[i, :],x), 0),
                                       MOI.LessThan(b[i]))
                end

                # Solve corresponding linear program
                MOI.optimize!(optimizer)

                status = MOI.get(optimizer, MOI.TerminationStatus())
                if status == MOI.OPTIMAL
                    # Pull out optimal value and compute
                    w_sol = MOI.get(optimizer, MOI.VariablePrimal(), x)
                    value = (1-delta)*flow_u(rpd, a1, a2) + delta*w_sol

                    # Save hyperplane level and continuation promises
                    Cia[ia] = h1*value[1] + h2*value[2]
                    Wia[:, ia] = value
                else
                    Cia[ia] = -Inf
                end
            end

            # Action which pushes furthest in direction h_i
            astar = argmax(Cia)
            a1star, a2star = AS[astar, :]

            # Get hyperplane level and continuation value
            Cstar = Cia[astar]
            Wstar = Wia[:, astar]
            if Cstar > -1e15
                Cnew[ih] = Cstar
            else
                error("Failed to find feasible action/continuation pair")
            end

            # Update the points
            Z[:, ih] = (1-delta)*flow_u(rpd, a1star, a2star) + delta*[Wstar[1],
                        Wstar[2]]
        end

        # Update distance and iteration counter
        dist = maximum(abs, C - Cnew)
        iter += 1

        if verbose && mod(iter, nskipprint) == 0
            println("$iter\t$dist\t($_w1, $_w2)")
        end

        if iter >= maxiter
            @warn "Maximum Iteration Reached"
        end

        # Update hyperplane levels
        copyto!(C, Cnew)
    end


    # Given the H-representation `(H, C)` of the computed polytope of
    # equilibrium payoff profiles, we obtain its V-representation `vertices`
    # using Polyhedra.jl (it uses `plib` which was chosen for computations)
    p = polyhedron(hrep(H, C), plib)
    vr = vrep(p)
    pts = points(vr)  # Vector of Vectors

    # Reduce the number of vertices by rounding points to the tolerance
    tol_int = round(Int, abs(log10(tol))) - 1

    # Find vertices that are unique within tolerance level
    vertices = Matrix{TD}(undef, (length(pts), 2))
    for (i, pt) in enumerate(pts)
        vertices[i, :] = round.(pt, digits=tol_int)
    end
    vertices = unique(vertices, dims=1)

    return vertices
end
