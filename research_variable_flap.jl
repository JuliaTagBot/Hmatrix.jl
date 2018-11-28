using HCubature
using LinearAlgebra
using ProgressMeter
using Parameters
using JLD2

struct Index
    i::Int64
    j::Int64
end

function Base.:(==)(a::Index, b::Index)
    return a.i==b.i && a.j==b.j
end

function Base.:+(a::Index, b::Index)
    c = Index(a.i + b.i, a.j+b.j)
    return c
end
function Base.:-(a::Index, b::Index)
    c = Index(a.i - b.i, a.j - b.j)
    return c
end
function coor(a::Index, h::Float64)
    return [a.i*h, a.j*h]
end

function inrange(J::Index, N::Int64)
    if abs(J.i)>N || abs(J.j)>N
        return false
    end
    
    if J.i>0 && J.j>0
        return false
    end
    
    return true
end


@with_kw mutable struct Data
    d::Union{Nothing,Dict} = nothing
    dinv::Union{Nothing,Dict} = nothing
    R::Union{Nothing,Int32} = nothing
    loc::Union{Nothing,Array{Float64,2}} = nothing
    A::Union{Nothing,Array{Float64,2}} = nothing
    F::Union{Nothing,Array{Float64}} = nothing
    N::Union{Nothing,Int64} = nothing
    h::Union{Nothing,Float64} = nothing
    r::Union{Nothing,Float64} = nothing
end

NewData(N::Int64) = Data(N = N, h = 1/N, r = 0.3)

function Cfun(ρ::Function, s::Float64, r::Float64)
    hquadrature(x->ρ(x^(1/(2-2s))), 0, r^(2-2s))[1]*2π/(2-2s)
end

function ρfun(x::Union{Float64,Array{Float64}}, r::Float64)
    x = norm(x)/r
    1-(70*x^9-315*x^8+540*x^7-420*x^6+126*x^5)
end

function sfunc(x::Array{Float64})
    #     @show size(x)
    #     @show size([0;0])
    if x[1]<=0.0 && x[2]>=0.0
        d = minimum([-x[1] x[1]+1.0 1.0-x[2]])
    elseif x[1]<=0.0 && x[2]<=0.0
        d = minimum([1.0+x[1] x[2]+1.0 norm(x-[0;0])])
    elseif x[1]>=0 && x[2]<=0.0
        d = minimum([1.0+x[2] -x[2] 1.0-x[1]])
    else
        return NaN
    end
    return 0.5+(1.0-2d)*0.4
end

function compute_vec(I::Index, data::Data)
    s = sfunc
    C = Cfun
    d = data.d
    N, h, r = data.N, data.h, data.r
    ρ = x->ρfun(norm(x), r)
    V = Dict([x[1]=>0.0 for x in d])
#     println(V)
    n = Int(ceil(r/h))
    x = coor(I, h)
    for i = -2N:2N
        for j = -2N:2N
            Y = Index(i,j)
            y = coor(Y, h)
            J = Y+I
            if inrange(J, N) && !(i==0 && j==0)
                V[J] += 1/norm(y)^(2+2*s(x))*h^2
            end
        end
    end
   
    res = 0.0
    for i = -5N:5N
        for j = -5N:5N
            y = coor(Index(i,j), h)
            (!(i==0 && j==0)) && (res -= 1/norm(y)^(2+2s(x))*h^2)
        end
    end
    V[I] += res
    
    res = 0.0
    for i = -n:n
        for j = -n:n
            y = coor(Index(i,j), h)
            (!(i==0 && j==0)) && (res -= 1/4 * ρ(y) / norm(y)^(2s(x))*h^2)
        end
    end
    res += 1/4 * C(ρ, s(x), r)
    
    for (i,j) in [(1,0), (-1,0), (0,1), (0,-1)]
        Y = Index(i,j) + I
        if inrange(Y, N)
            V[Y] += res/h^2
        end
    end
    V[I] -= 4res/h^2
    
    return V
end



function index2idx(N::Int64)
    d = Dict{Index, Int64}()
    idx = 1
    for i = -N:N
        for j = -N:N
            Y = Index(i, j)
            if inrange(Y, N)
                d[Y] = idx
                idx += 1
            end
        end
    end
    return d
end


using PyPlot
using PyCall
@pyimport scipy.interpolate as spitp

NN = 300
xs = LinRange(-1,1,NN)
Xs = zeros(NN,NN); Ys = zeros(NN,NN)
for i = 1:NN
    for j = 1:NN
        Xs[i,j] = xs[i]
        Ys[i,j] = xs[j]
    end
end
function showsol(u, loc)
    Zs = spitp.griddata((loc[:,1], loc[:,2]), u, (Xs, Ys))
    for i = 1:NN
        for j = 1:NN
            if Xs[i,j]>0 && Ys[i,j]>0
                Zs[i,j] = NaN
            end
        end
    end
    pcolor(Xs, Ys, Zs, vmin=0.0, vmax=0.038)
#     contour(X,Y,Z)
    colorbar()
    xlabel("x")
    ylabel("y")
end

function computeA(data::Data)
    N, r, h = data.N, data.r, data.h
    d = index2idx(N)
    dinv = Dict()
    R = length(d)
    loc = zeros(R, 2)
    A = zeros(R, R)
    
    for i = -N:N
        for j = -N:N
            if inrange(Index(i,j), N)
                v = compute_vec(Index(i,j), data);
                loc[d[Index(i,j)],:] = coor(Index(i,j), h)
                for x in v
                    A[d[Index(i,j)], d[x[1]]] += x[2]
                    dinv[d[Index(i,j)]] = Index(i,j)
                end
            end
        end
    end
    data.d, data.dinv, data.R, data.loc = d, dinv, R, loc;
    A = -A; # to obtain -(-Δ)^(2s(x))
    data.A = A
end

function split_into_n_pieces(N::Int64, n::Int64)
    m = Int(floor(N/n))
    P = zeros(Int64, n+1)
    P[1] = 0
    for i = 1:n-1
        P[i+1] = i*m
    end
    P[end] = N
    return P
end

function computeA_parallel()
    # others
    N, r, h = data.N, data.r, data.h
    d = index2idx(N)
    dinv = Dict()
    R = length(d)
    loc = zeros(R, 2)
    A = zeros(R, R)
    for i = -N:N
        for j = -N:N
            if inrange(Index(i,j), N)
                loc[d[Index(i,j)],:] = coor(Index(i,j), h)
                dinv[d[Index(i,j)]] = Index(i,j)
            end
        end
    end
    data.d, data.dinv, data.R, data.loc = d, dinv, R, loc;
    
    ######################### PARALLEL ############################
    c = RemoteChannel(()->Channel{Array{Float64}}(nprocs()))
#     c = RemoteChannel(()->Channel{Int64}(nprocs()))
    function rcall()       
        N, r, h = data.N, data.r, data.h
        d = index2idx(N)
        R = length(d)
        @assert R!=nothing
        A = zeros(R, R)
        K = [x[1] for x in d]
        P = split_into_n_pieces(length(K), nprocs()-1)
        p = myid()
        println("Process $p ==> ",P[p-1]+1:P[p])
        for q = P[p-1]+1:P[p]
            I = K[q]
            v = compute_vec(I, data);
            for x in v
                A[d[I], d[x[1]]] += x[2]
            end
        end
        return A
    end
    
    for p = 2:nprocs()
        @async put!(c, remotecall_fetch(rcall, p))
    end
    
    B = zeros(data.R, data.R)
    println("          |",reduce(*, ["-" for i = 2:nprocs()]),"|")
    print("Processing|")
    t = @timed for i = 2:nprocs()
        A = take!(c)
        B += A
        print("*")
    end
    println("|",t[2]," sec","(avg ", t[2]/(nprocs()-1)," sec)")
    B = -B; # to obtain -(-Δ)^(2s(x))
    data.A = B
end

function fsrc(x::Array{Float64})
    d1 = norm(x-[-0.5;0.5])
    d2 = norm(x-[0.5;-0.5])
    d3 = norm(x-[-0.5;-0.5])
    return exp(-10*d1^2)+exp(-10*d2^2)+exp(-10*d3^2)
end

function computeF(data::Data)
    R, N, h, d = data.R, data.N, data.h, data.d
    R = length(data.d)
    F = zeros(R)
    for i = -N:N
        for j = -N:N
            if inrange(Index(i,j), N)
                x = coor(Index(i,j), h)
                F[d[Index(i,j)]] = fsrc(x)
            end
        end
    end
    data.F = F;
end

