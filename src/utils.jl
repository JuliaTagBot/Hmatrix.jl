export
Hparams,
matshow,
plot,
coarsening

@with_kw mutable struct Params
    Geom::Union{Nothing, Array} = nothing
    εComp::Float64 = 1e-10
    MaxRank::Int64 = 16
    MaxBlock::Int64 = -1
    MinBlock::Int64 = 64
    Kernel::Union{Function, Nothing} = nothing
    α::Union{Array{Function,1}, Nothing} = nothing
    β::Union{Array{Function,1}, Nothing} = nothing
    CompMethod::String = "svd"
    ConsMethod::String = "bbfmm"
    εTrunc::Float64 = 1e-10
    verbose::Bool = false
    η::Float64 = 0.55 # admissible parameter
    aca_force::Bool = false # if true, aca will not check validity
end

Hparams = Params()


function coarsening(H::Hmat)
    G = copy(H)
    function helper(H::Hmat)
        if (H.s!=H.t) && H.is_fullmatrix==true
            C = H.C
            if size(C,1)==size(C,2)
                U,S,V = psvd(C)    # fast svd is availabel
            else
                U,S,V = svd(C)
            end
            k = findlast(S/S[1] .> Hparams.εTrunc)
            # @show k, Hparams.MaxRank
            if k<=Hparams.MaxRank
                H.A = U[:,1:k]
                H.B = Array((diagm(0=>S[1:k])*V'[1:k,:])')
                # @show H.m, H.n, size(H.A), size(H.B)
                H.C = zeros(0,0)
                H.is_rkmatrix = true
                H.is_fullmatrix = false
            end
        elseif H.is_hmat
            for i = 1:2
                for j = 1:2
                    helper(H.children[i,j])
                end
            end
        end
    end
    helper(G)
    G
end

function require_args(args...)
    for i = 1:length(args)
        if args[i]==nothing
            error("Argument $i is nothing")
        end
    end
end

function Base.:size(H::Hmat)
    return (H.m, H.n)
end

function Base.:size(H::Hmat, i::Int64)
    if i==1
        return H.m
    elseif i==2
        return H.n
    else
        @error "size(Hmat, Int64): invalid dimension $i"
    end
end

function info(H::Hmat)
    dmat::Int64 = 0
    rkmat::Int64 = 0
    level::Int64 = 0
    compress_ratio::Float64 = 0
    function helper(H::Hmat, l::Int)
        # global dmat
        # global rkmat
        # global level
        if H.is_fullmatrix
            dmat += 1
            level = max(l, level)
            compress_ratio += H.m*H.n
        elseif H.is_rkmatrix
            rkmat += 1
            level = max(l, level)
            compress_ratio += size(H.A,1)*size(H.A,2) + size(H.B,1)*size(H.B, 2)
        else
            for i = 1:2
                for j = 1:2
                    helper(H.children[i,j], l+1)
                end
            end
        end
    end
    helper(H, 1)
    return dmat, rkmat, level, compress_ratio/H.m/H.n
end

function rkmat(A, B, s, t)
    return Hmat(m = size(A,1), n = size(B,1), s=s, t=t, is_rkmatrix = true, A = A, B = B)
end

function fullmat(A, s, t)
    return Hmat(m = size(A,1), n = size(A,2), s = s, t = t, is_fullmatrix = true, C = A)
end

function hmat_from_children(h11, h12, h21, h22, s, t)
    H = Hmat(m = h11.m + h21.m, n = h11.n + h12.n, s=s, t = t, is_hmat = true)
    H.children = [h11 h12
                h21 h22]
    return H
end



# copy the hmatrix A to H in place.
function hmat_copy!(H::Hmat, A::Hmat)
    H.m = A.m
    H.n = A.n
    H.s = A.s
    H.t = A.t
    H.P = copy(A.P)
    if A.is_fullmatrix
        H.C = copy(A.C) # bottleneck
        H.is_fullmatrix = true
    elseif A.is_rkmatrix
        H.A = copy(A.A)
        H.B = copy(A.B)
        H.is_rkmatrix = true
    else
        H.is_hmat = true
        H.children = Array{Hmat}([Hmat() Hmat()
                                Hmat() Hmat()])
        for i = 1:2
            for j = 1:2
                hmat_copy!(H.children[i,j], A.children[i,j])
            end
        end
    end
end

function Base.:copy(H::Hmat)
    G = Hmat()
    hmat_copy!(G, H)
    return G
end

# convert matrix A to full matrix
function to_fmat!(A::Hmat)
    if A.is_fullmatrix
        return
    elseif A.is_rkmatrix
        A.C = A.A*A.B'
        A.A, A.B = zeros(0,0), zeros(0,0)
    elseif A.is_hmat
        for i = 1:2
            for j = 1:2
                to_fmat!(A.children[i,j])
            end
        end
        A.C = [A.children[1,1].C A.children[1,2].C
                A.children[2,1].C  A.children[2,2].C]
        if length(A.children[1,1].P)>0
            A.P = [A.children[1,1].P;
                    A.children[2,2].P .+ A.children[1,1].m]
        end
        A.children = Array{Hmat}([])
    end
    A.is_fullmatrix = true
    A.is_rkmatrix = false
    A.is_hmat = false
end

function to_fmat(A::Hmat)
    B = copy( A)
    to_fmat!(B)
    return B.C
end
Base.:Array(A::Hmat) = to_fmat(A)

function add_patch(a,b,c,d, n; edgecolor, color)
    return patch.Rectangle([c,a], d-c, b-a, edgecolor=edgecolor, color=color, alpha=0.6)
end

function PyPlot.:matshow(H::Hmat)
    cfig = figure()
    ax = cfig[:add_subplot](1,1,1)
    ax[:set_aspect]("equal")
    function helper(H)
        if H.is_fullmatrix
            c = add_patch( H.s.s, H.s.e, H.t.s, H.t.e, H.m, edgecolor="k", color="y")
            ax[:add_artist](c)
        elseif H.is_rkmatrix
            c = add_patch( H.s.s, H.s.e, H.t.s, H.t.e, H.m, edgecolor="k", color="g")
            ax[:add_artist](c)
        else
            for i = 1:2
                for j = 1:2
                    helper(H.children[i,j])
                end
            end
        end
    end
    helper(H)
    xlim([1,H.m])
    ylim([1,H.n])
    gca()[:invert_yaxis]()
    show()
end

function Base.:print(c::Cluster;verbose=false)
    current_level = [c]
    while length(current_level)>0
        if verbose
            println(join(["$(x.N)($(x.s),$(x.e))" for x in current_level], " "))
        else
            println(join(["$(x.N)" for x in current_level], " "))
        end
        next_level = []
        for n in current_level
            if !n.isleaf 
                push!(next_level, n.left)
                push!(next_level, n.right)
            end
            current_level = next_level
        end
    end
end

function Base.:print(h::Hmat)
    G = to_fmat(h)
    printmat(G)
end

function PyPlot.:plot(c::Cluster; showit=false)
    
    function helper(level)
        clevel = 0
        current_level = [c]
        while clevel!=level && length(current_level)>0
            next_level = []
            for n in current_level
                if !n.isleaf 
                    push!(next_level, n.left)
                    push!(next_level, n.right)
                end
                current_level = next_level
            end
            clevel += 1
        end
        if length(current_level)==0
            return false
        end
        figure()
        for c in current_level
            if size(c.X,2)==1
                scatter(c.X[:,1], ones(size(c.X,1)), marker = ".")
            elseif size(c.X, 2)==2
                scatter(c.X[:,1], c.X[:,2], marker = ".")
            elseif size(c.X,2)==3
                scatter3D(c.X[:,1], c.X[:,2], c.X[:,3], marker = ".")
            end
        end
        title("Level = $level")
        if !showit
            savefig("level$level.png")
            close("all")
        end
        return true
    end
    flag = true
    l = 0
    while flag
        flag = helper(l)
        l += 1
    end
end


function printmat(H)
    println("=============== size = $(size(H,1))x$(size(H,2)) ===================")
    for i = 1:size(H,1)
        for j = 1:size(H,2)
            @printf("%+0.4f ", H[i,j])
        end
        @printf("\n")
    end
    println("=====================================================================")
end

function check_if_equal(H::Hmat, C::Array{Float64})
    G = (H)
    to_fmat!(G)
    println("Error = $(norm(C-G.C,2)/norm(C,2))")
end

function verify_matrix_error(H::Hmat, C::Array{Float64})
    G = to_fmat(H)
    err = pointwise_error(G, C)
    println("Matrix Error = $err")
end

function verify_matvec_error(H::Hmat, C::Array{Float64})
    y = rand(size(C,1))
    b1 = H*y
    b2 = C*y
    err = norm(b2-b1)/norm(b2)
    println("Matvec Error = $err")
end

function verify_lu_error(H1::Hmat, H::Hmat, A::Array{Float64})
    C = to_fmat(H1)
    HC = to_fmat(H)
    U = UpperTriangular(HC)
    L = (LowerTriangular(HC)-diagm(0=>diag(HC))+UniformScaling(1.0))
    x = rand(size(A,1))
    b = C*x
    # x = U\(L\b[H.P])
    y = H\b
    err1 = norm(x-y)/norm(x)
    # println("Permuation = $(H.P)")
    
    G = C[H.P,:] - L*U
    println("[Operator] LU Error = $(maximum(abs.(G)))")

    G = A[H.P,:] - L*U
    println("[Matrix  ] LU Error = $(maximum(abs.(G)))")
    x = rand(size(A,1))
    b = A*x
    y = H\b
    err = norm(x-y)/norm(x)
    println("[Operator] Solve Error = $err1")
    println("[Matrix  ] Solve Error = $err")
    
end


function verify_lu_error(HH::Hmat; A = nothing)
    H = copy(HH)
    C = to_fmat(H)
    lu!(H)
    x = rand(size(C,1))
    b = C*x
    y = H\b
    err = norm(x-y)/norm(x)
    # println("Permuation = $(H.P)")
    println("Solve Error = $err")
    to_fmat!(H)
    G = C[H.P,:] - (LowerTriangular(H.C)-diagm(0=>diag(H.C))+UniformScaling(1.0))*UpperTriangular(H.C)
    println("LU Operator Error = $(maximum(abs.(G)))")

    
    if A!=nothing
        G = A[H.P,:] - (LowerTriangular(H.C)-diagm(0=>diag(H.C))+UniformScaling(1.0))*UpperTriangular(H.C)
        println("LU Matrix Error = $(maximum(abs.(G)))")
        x = rand(size(A,2))
        b = A*x
        y = H\b
        err = norm(x-y)/norm(x)
        # println("Permuation = $(H.P)")
        println("Matrix Solve Error = $err")
    end
    return G
end

function check_err(HH::Hmat, C::Array{Float64})
    H = copy(HH)
    to_fmat!(H)
    # println(H.P)
    G = C[H.P,:] - (LowerTriangular(H.C)-diagm(0=>diag(H.C))+UniformScaling(1.0))*UpperTriangular(H.C)
    println("Matrix Error = $(maximum(abs.(G)))")
end

function pointwise_error(A, B)
    return maximum(abs.(A-B))
end

function rel_error(x::Array{Float64}, y::Array{Float64})
    return norm(y-x)/norm(y)
end