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

# ---------- Case: basis_roundtrip (cross-language JSON) ----------
let
    Random.seed!(42)
    m, n = 2, 2
    code_fwd, tensors_raw = qft_code(m, n)
    tensors = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_raw]
    # Perturb tensors slightly so they differ from defaults (meaningful roundtrip)
    perturbed = [t .+ ComplexF64(1e-6 * (0.3 + 0.5im)) for t in tensors]
    basis = QFTBasis(m, n, perturbed)

    # Save to JSON using upstream's save_basis
    json_path = joinpath(OUT_DIR, "qft_basis_trained.json")
    save_basis(json_path, basis)

    # Also save expected ft_mat output on a fixed image for verification
    pic = Complex{Float64}.(rand(2^m, 2^n))
    fwd = ft_mat(basis.tensors, code_fwd, m, n, pic)
    expected_hash = ParametricDFT.basis_hash(basis)
    npzwrite(joinpath(OUT_DIR, "basis_roundtrip.npz"),
             Dict("pic" => pic,
                  "fwd" => fwd,
                  "m" => [m], "n" => [n]))
    # Record hash in a side file so Python tests can check it independently of JSON parse
    open(joinpath(OUT_DIR, "basis_roundtrip_hash.txt"), "w") do io
        write(io, expected_hash)
    end
end

# ---------- Case: adam_trajectory_4x4 ----------
let
    Random.seed!(0)
    m, n = 2, 2
    code_fwd, tensors_init_raw = qft_code(m, n)
    tensors_init = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_init_raw]
    target = Complex{Float64}.(rand(2^m, 2^n))

    loss_fn(ts) = ParametricDFT.loss_function(ts, m, n, code_fwd, target, L1Norm())
    grad_fn(ts) = Zygote.gradient(loss_fn, ts)[1]

    loss_trace = Float64[]
    push!(loss_trace, Float64(loss_fn(tensors_init)))

    final = optimize!(RiemannianAdam(lr=0.01), copy(tensors_init), loss_fn, grad_fn;
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
    npzwrite(joinpath(OUT_DIR, "adam_trajectory_4x4.npz"), kv)
end

# ---------- Case: entangled_qft_4x4 ----------
let
    Random.seed!(8)
    m, n = 2, 2
    entangle_phases = [0.3, 0.5]
    code_fwd, tensors_raw, n_ent = entangled_qft_code(m, n; entangle_phases=entangle_phases)
    tensors = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_raw]
    pic = Complex{Float64}.(rand(2^m, 2^n))
    fwd = reshape(code_fwd(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
    kw = Dict{String, Any}("pic" => pic, "fwd" => fwd, "n_entangle" => [n_ent],
                           "entangle_phases" => entangle_phases)
    for (i, t) in enumerate(tensors)
        kw["tensor_$(i-1)"] = collect(t)
    end
    npzwrite(joinpath(OUT_DIR, "entangled_qft_4x4.npz"), kw)
end

# ---------- Case: tebd_4x4 ----------
let
    Random.seed!(9)
    m, n = 2, 2
    phases = [0.1, 0.2, 0.3, 0.4]
    code_fwd, tensors_raw, _, _ = tebd_code(m, n; phases=phases)
    tensors = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_raw]
    pic = Complex{Float64}.(rand(2^m, 2^n))
    fwd = reshape(code_fwd(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
    kw = Dict{String, Any}("pic" => pic, "fwd" => fwd, "phases" => phases)
    for (i, t) in enumerate(tensors)
        kw["tensor_$(i-1)"] = collect(t)
    end
    npzwrite(joinpath(OUT_DIR, "tebd_4x4.npz"), kw)
end

# ---------- Case: mera_4x4 ----------
let
    Random.seed!(11)
    m, n = 2, 2
    phases = [0.1, 0.2, 0.3, 0.4]
    code_fwd, tensors_raw, _, _ = mera_code(m, n; phases=phases)
    tensors = Matrix{ComplexF64}[Matrix{ComplexF64}(t) for t in tensors_raw]
    pic = Complex{Float64}.(rand(2^m, 2^n))
    fwd = reshape(code_fwd(tensors..., reshape(pic, fill(2, m+n)...)), 2^m, 2^n)
    kw = Dict{String, Any}("pic" => pic, "fwd" => fwd, "phases" => phases)
    for (i, t) in enumerate(tensors)
        kw["tensor_$(i-1)"] = collect(t)
    end
    npzwrite(joinpath(OUT_DIR, "mera_4x4.npz"), kw)
end

# ---------- Case: compression_roundtrip_4x4 ----------
let
    Random.seed!(17)
    m, n = 2, 2
    basis = QFTBasis(m, n)
    img = rand(2^m, 2^n)
    k = 8
    compressed = ParametricDFT.compress_with_k(basis, img; k=k)
    recovered = ParametricDFT.recover(basis, compressed; verify_hash=true)
    # Save: original image, kept indices (1-based column-major), real/imag,
    # k, and the recovered image for direct Python comparison.
    npzwrite(joinpath(OUT_DIR, "compression_roundtrip_4x4.npz"),
             Dict("image" => img,
                  "k" => [k],
                  "indices" => compressed.indices,
                  "values_real" => compressed.values_real,
                  "values_imag" => compressed.values_imag,
                  "recovered" => recovered,
                  "basis_hash_chars" => collect(UInt8, compressed.basis_hash)))
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
        for f in readdir(OUT_DIR)
        if endswith(f, ".npz") || endswith(f, ".json") || endswith(f, ".txt")
    ),
)
open(joinpath(OUT_DIR, "manifest.json"), "w") do io
    JSON3.pretty(io, manifest)
end

println("Goldens written to $OUT_DIR")
