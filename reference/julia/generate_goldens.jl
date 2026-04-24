# Reference-vector generator. Run from repo root:
#   julia --project=reference/julia reference/julia/generate_goldens.jl
#
# Writes reference/goldens/*.npz and reference/goldens/manifest.json.
using Pkg

const UPSTREAM_SHA = "a201a27e47df2f0f3ab460f83d49b6e5f5d1e9ef"
const HARNESS_DIR = @__DIR__
const OUT_DIR = normpath(joinpath(HARNESS_DIR, "..", "goldens"))
mkpath(OUT_DIR)

# Install pinned upstream on first run; subsequent runs are no-ops.
deps = Pkg.dependencies()
if !any(p.name == "ParametricDFT" for p in values(deps))
    Pkg.add(url="https://github.com/nzy1997/ParametricDFT.jl", rev=UPSTREAM_SHA)
end

using ParametricDFT
using NPZ
using Random
using SHA
using JSON3
using LinearAlgebra
using Dates
using Zygote

# ---------- Deterministic RNG ----------
Random.seed!(0)

# ---------- Case: qft_code_4x4 ----------
let
    code, tensors = qft_code(2, 2)
    kw = Dict{String, Any}("n_tensors" => length(tensors))
    for (i, t) in enumerate(tensors)
        kw["tensors_$(i-1)"] = collect(t)
    end
    npzwrite(joinpath(OUT_DIR, "qft_code_4x4.npz"), kw)
end

# ---------- Case: ft_mat_roundtrip ----------
let
    Random.seed!(123)
    pic = Complex{Float64}.(rand(4, 4))
    code_fwd, tensors = qft_code(2, 2)
    code_inv, _       = qft_code(2, 2; inverse=true)
    fwd = ft_mat(tensors, code_fwd, 2, 2, pic)
    roundtrip = ift_mat(conj.(tensors), code_inv, 2, 2, fwd)
    npzwrite(joinpath(OUT_DIR, "ft_mat_roundtrip.npz"),
             Dict("pic" => pic, "fwd" => fwd, "roundtrip" => roundtrip))
end

# ---------- Case: manifold_project_retract ----------
let
    Random.seed!(7)
    d, n = 4, 3
    A = randn(ComplexF64, d, d, n)
    U = similar(A)
    for k in 1:n
        Q, _ = qr(A[:, :, k]); U[:, :, k] = Matrix(Q)
    end
    G = randn(ComplexF64, d, d, n)
    UM = ParametricDFT.UnitaryManifold()
    Xi = ParametricDFT.project(UM, U, G)
    U_new = ParametricDFT.retract(UM, U, Xi, 0.1)
    npzwrite(joinpath(OUT_DIR, "manifold_project_retract.npz"),
             Dict("U" => U, "G" => G, "Xi" => Xi, "U_new" => U_new, "alpha" => [0.1]))
end

# ---------- Case: loss_values ----------
let
    Random.seed!(5)
    pred   = randn(ComplexF64, 4, 4)
    target = randn(ComplexF64, 4, 4)
    l1 = sum(abs.(pred))
    topk_1 = ParametricDFT.topk_truncate(pred, 1)
    topk_3 = ParametricDFT.topk_truncate(pred, 3)
    topk_5 = ParametricDFT.topk_truncate(pred, 5)
    mse_all = sum(abs2.(pred .- target))
    npzwrite(joinpath(OUT_DIR, "loss_values.npz"),
             Dict("pred" => pred, "target" => target,
                  "l1" => [l1], "mse" => [mse_all],
                  "topk_1" => topk_1, "topk_3" => topk_3, "topk_5" => topk_5))
end

# ---------- Case: train_trajectory_4x4 ----------
let
    Random.seed!(0)
    m, n = 2, 2
    code_fwd, tensors_init_raw = qft_code(m, n)
    code_inv, _                = qft_code(m, n; inverse=true)
    # Ensure concrete Matrix{ComplexF64} eltype so it satisfies
    # Vector{<:AbstractMatrix} (optimize!'s required signature).
    tensors_init = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_init_raw]
    target = Complex{Float64}.(rand(2^m, 2^n))

    loss_fn(ts) = ParametricDFT.loss_function(ts, m, n, code_fwd, target, L1Norm())
    grad_fn(ts) = Zygote.gradient(loss_fn, ts)[1]

    loss_trace = Float64[]
    push!(loss_trace, Float64(loss_fn(tensors_init)))

    final = optimize!(RiemannianGD(lr=0.01), copy(tensors_init), loss_fn, grad_fn;
                      max_iter=50, tol=1e-10, loss_trace=loss_trace)

    kv = Dict{String, Any}(
        "target" => target,
        "loss_history" => loss_trace,
        "config_lr" => [0.01], "config_steps" => [50], "config_seed" => [0],
    )
    for (i, t) in enumerate(tensors_init)
        kv["tensors_init_$(i-1)"] = collect(t)
    end
    for (i, t) in enumerate(final)
        kv["tensors_final_$(i-1)"] = collect(t)
    end
    npzwrite(joinpath(OUT_DIR, "train_trajectory_4x4.npz"), kv)
end

# ---------- Manifest ----------
function file_sha256(path)
    open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

manifest = Dict(
    "upstream_sha" => UPSTREAM_SHA,
    "julia_version" => string(VERSION),
    "generated_at" => string(now()),
    "files" => Dict(
        f => file_sha256(joinpath(OUT_DIR, f))
        for f in readdir(OUT_DIR) if endswith(f, ".npz")
    ),
)
open(joinpath(OUT_DIR, "manifest.json"), "w") do io
    JSON3.pretty(io, manifest)
end

println("Goldens written to $OUT_DIR")
