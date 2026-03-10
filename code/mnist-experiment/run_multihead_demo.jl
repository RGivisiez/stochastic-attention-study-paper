#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Multi-Head Stochastic Attention Demo (Q5 response)
#
# Shows that SA extends naturally to multi-head attention using PCA-partitioned
# subspaces as a proxy for learned projections.
#
# Key insight: with K patterns in d dimensions, the data spans a K-dimensional
# subspace. We partition the top-K principal components across H heads, so each
# head captures different variance directions of the memory.
#
# Configurations:
#   1. Single-head SA (H=1, full d=784) at β=200 (generation)
#   2. Multi-head SA (H=2, 50 PCs/head) at β=200
#   3. Multi-head SA (H=4, 25 PCs/head) at β=200
#   4. Multi-head SA (H=5, 20 PCs/head) at β=200
#   5. Multi-head SA (H=4, 25 PCs/head) at β=2000 (retrieval)
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO, Printf
flush(stdout); flush(stderr)
@info "Environment loaded."

# ── Parameters ───────────────────────────────────────────────────────────────
const number_of_examples  = 100
const number_of_rows      = 28
const number_of_cols      = 28
const number_of_pixels    = number_of_rows * number_of_cols  # d = 784
const α_step              = 0.01
const β_gen               = 200.0
const β_ret               = 2000.0
const n_chains            = 30
const T_per_chain         = 5000
const T_burnin            = 2000
const thin_interval       = 100
const samples_per_chain   = 5
const σ_init              = 0.01
const DIGIT               = 3

# ── Helpers ──────────────────────────────────────────────────────────────────
function decode(s::Vector{<:Number}; nr::Int=28, nc::Int=28)
    X = reshape(s, nr, nc) |> X -> transpose(X) |> Matrix
    return replace(X, -1 => 0)
end

function build_grid(samples; nrows=4, ncols=4, gap=2)
    H, W    = 28, 28
    canvas  = zeros(Float64, nrows*H + (nrows-1)*gap, ncols*W + (ncols-1)*gap)
    indices = round.(Int, range(1, length(samples), length=nrows*ncols))
    for idx in 1:(nrows*ncols)
        r  = div(idx-1, ncols);  c  = rem(idx-1, ncols)
        y0 = r*(H+gap)+1;        x0 = c*(W+gap)+1
        img = decode(samples[indices[idx]])
        lo, hi = minimum(img), maximum(img)
        hi > lo && (img = (img .- lo) ./ (hi - lo))
        canvas[y0:y0+H-1, x0:x0+W-1] .= img
    end
    return canvas
end

function chain_metric_se(samps, metric_fn; nc=n_chains, spc=samples_per_chain)
    vals = [metric_fn(samps[(i-1)*spc+1:i*spc]) for i in 1:nc]
    return std(vals) / sqrt(nc)
end

# ── Load MNIST ───────────────────────────────────────────────────────────────
@info "Loading MNIST …"
digits_dict = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)
@info "MNIST loaded."

# ── Build normalised memory matrix (d × K) ──────────────────────────────────
ϵ_norm = 1e-12
X_raw = zeros(Float64, number_of_pixels, number_of_examples)
X̂     = zeros(Float64, number_of_pixels, number_of_examples)
for i in 1:number_of_examples
    xᵢ = reshape(transpose(digits_dict[DIGIT][:, :, i]) |> Matrix, number_of_pixels) |> vec
    X_raw[:, i] = xᵢ
end
for i in 1:number_of_examples
    X̂[:, i] = X_raw[:, i] ./ (norm(X_raw[:, i]) + ϵ_norm)
end
K = size(X̂, 2)
d = size(X̂, 1)
@info "Memory matrix: $(size(X̂))  (d=$d, K=$K)"
flush(stdout); flush(stderr)

# ══════════════════════════════════════════════════════════════════════════════
# Compute PCA of the memory matrix
# ══════════════════════════════════════════════════════════════════════════════
@info "Computing PCA of memory matrix …"
X_mean = mean(X̂, dims=2) |> vec
X_centered = X̂ .- X_mean
F = svd(X_centered)
# F.U[:, 1:K] spans the data subspace (K non-zero singular values)
n_pcs = min(K, size(F.U, 2))  # number of meaningful PCs
PC = F.U[:, 1:n_pcs]  # d × n_pcs
@info "PCA done. $n_pcs meaningful PCs. Top 5 singular values: $(round.(F.S[1:5], digits=2))"
total_var = sum(F.S[1:n_pcs] .^ 2)
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# Multi-head SA: partition top-K PCs across heads
# ══════════════════════════════════════════════════════════════════════════════

"""
Run multi-head SA where each head gets n_pcs÷H principal components.
Head 1 gets PCs 1..d_head (highest variance), head H gets the lowest.
"""
function multihead_pca_sa(X̂::Matrix{Float64}, PC::Matrix{Float64},
    X_mean::Vector{Float64}, n_heads::Int;
    n_chains::Int=30, T::Int=5000, T_burnin::Int=2000,
    thin::Int=100, spc::Int=5, β::Float64=200.0, α::Float64=0.01)

    d, K = size(X̂)
    n_pcs = size(PC, 2)
    pcs_per_head = n_pcs ÷ n_heads
    @assert pcs_per_head * n_heads <= n_pcs "n_pcs=$n_pcs not divisible by H=$n_heads"

    all_samples = Vector{Vector{Float64}}()

    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))

    for (c, k) in enumerate(pattern_indices)
        chain_seed = 12345 + c

        # Run SA per head in its PCA subspace
        head_trajectories = Vector{Matrix{Float64}}(undef, n_heads)

        for h in 1:n_heads
            pc_start = (h-1) * pcs_per_head + 1
            pc_end   = h * pcs_per_head
            W_h = PC[:, pc_start:pc_end]'  # pcs_per_head × d

            # Project memories into head subspace
            M_h = W_h * X̂  # pcs_per_head × K
            for ki in 1:K
                nrm = norm(M_h[:, ki])
                nrm > ϵ_norm && (M_h[:, ki] ./= nrm)
            end

            # Project starting point
            Random.seed!(chain_seed)
            z₀ = W_h * (X̂[:, k] .+ σ_init .* randn(d))

            # Run SA chain in subspace
            (_, Ξ_h) = sample(M_h, z₀, T; β=β, α=α, seed=chain_seed + h)
            head_trajectories[h] = Ξ_h
        end

        # Collect thinned samples
        chain_pool = Vector{Vector{Float64}}()
        for tᵢ in (T_burnin+1):thin:T
            # Reconstruct: ξ = X_mean + Σ_h PC[:, range_h] * z_h(tᵢ)
            ξ_out = copy(X_mean)
            for h in 1:n_heads
                pc_start = (h-1) * pcs_per_head + 1
                pc_end   = h * pcs_per_head
                z_h = head_trajectories[h][tᵢ, :]
                ξ_out .+= PC[:, pc_start:pc_end] * z_h
            end
            push!(chain_pool, ξ_out)
        end

        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(spc, n_avail)))
        for idx in idxs
            push!(all_samples, chain_pool[idx])
        end
    end

    return all_samples
end

# ══════════════════════════════════════════════════════════════════════════════
# Run experiments
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Single-head SA at β=200 ───────────────────────────────────────────────
@info "Running single-head SA (H=1) at β=$β_gen …"
flush(stdout)
sa1_samples = Vector{Vector{Float64}}()
Random.seed!(42)
pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
for (c, k) in enumerate(pattern_indices)
    Random.seed!(12345 + c)
    sₒ = X̂[:, k] .+ σ_init .* randn(number_of_pixels)
    (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=β_gen, α=α_step, seed=12345 + c)
    chain_pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
    n_avail = length(chain_pool)
    idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
    for idx in idxs; push!(sa1_samples, chain_pool[idx]); end
end
@info "H=1: $(length(sa1_samples)) samples"
flush(stdout)

# ── 2-4. Multi-head SA at β=200 with varying H ──────────────────────────────
head_configs = [2, 4, 5]  # 100/2=50, 100/4=25, 100/5=20 PCs per head
mh_samples_gen = Dict{Int, Vector{Vector{Float64}}}()

for H in head_configs
    d_head = n_pcs ÷ H
    @info "Running multi-head PCA SA (H=$H, $d_head PCs/head) at β=$β_gen …"
    flush(stdout)
    mh_samples_gen[H] = multihead_pca_sa(X̂, PC, X_mean, H;
        n_chains=n_chains, T=T_per_chain, T_burnin=T_burnin,
        thin=thin_interval, spc=samples_per_chain, β=β_gen, α=α_step)
    @info "H=$H gen: $(length(mh_samples_gen[H])) samples"
    flush(stdout)
end

# ── 5. Multi-head SA (H=4) at β=2000 (retrieval) ────────────────────────────
@info "Running multi-head PCA SA (H=4) at β=$β_ret (retrieval) …"
flush(stdout)
sa4_ret = multihead_pca_sa(X̂, PC, X_mean, 4;
    n_chains=n_chains, T=T_per_chain, T_burnin=T_burnin,
    thin=thin_interval, spc=samples_per_chain, β=β_ret, α=α_step)
@info "H=4 ret: $(length(sa4_ret)) samples"
flush(stdout)

# ══════════════════════════════════════════════════════════════════════════════
# Compute and display metrics
# ══════════════════════════════════════════════════════════════════════════════

methods = [
    "SA (H=1, β=200)"       => sa1_samples,
    "SA (H=2, β=200)"       => mh_samples_gen[2],
    "SA (H=4, β=200)"       => mh_samples_gen[4],
    "SA (H=5, β=200)"       => mh_samples_gen[5],
    "SA (H=4, β=2000)"      => sa4_ret,
]

println("\n" * "="^80)
println("MULTI-HEAD STOCHASTIC ATTENTION DEMO — RESULTS")
println("="^80)
println("PCA-partitioned subspaces over top-$n_pcs PCs, K=$K digit-$DIGIT")
println("30 chains × 5000 steps, 2000 burn-in, thin/100, 5/chain = 150 total")
println("="^80)

println("\n%----- Table: Multi-Head SA Comparison -----")
println("Method & Novelty & Diversity & Energy & Max-Cos \\\\")
println("\\midrule")

for (name, samps) in methods
    nov_vals = [sample_novelty(ξ, X̂) for ξ in samps]
    nov_mean = mean(nov_vals)
    nov_se   = chain_metric_se(samps, g -> mean(sample_novelty(ξ, X̂) for ξ in g))

    div_mean = sample_diversity(samps)
    div_se   = chain_metric_se(samps, g -> sample_diversity(g))

    en_vals  = [hopfield_energy(ξ, X̂, β_gen) for ξ in samps]
    en_mean  = mean(en_vals)
    en_se    = chain_metric_se(samps, g -> mean(hopfield_energy(ξ, X̂, β_gen) for ξ in g))

    maxcos_vals = [1.0 - sample_novelty(ξ, X̂) for ξ in samps]
    maxcos_mean = mean(maxcos_vals)
    maxcos_se   = chain_metric_se(samps, g -> mean(1.0 - sample_novelty(ξ, X̂) for ξ in g))

    println("$name & " *
            "$(round(nov_mean, digits=3)) ± $(round(nov_se, digits=3)) & " *
            "$(round(div_mean, digits=3)) ± $(round(div_se, digits=3)) & " *
            "$(round(en_mean, digits=1)) ± $(round(en_se, digits=1)) & " *
            "$(round(maxcos_mean, digits=3)) ± $(round(maxcos_se, digits=3)) \\\\")
end

# ── Per-head variance explained ──────────────────────────────────────────────
println("\n%----- Variance explained per head (H=4, 25 PCs each) -----")
pcs_per_head_4 = n_pcs ÷ 4
for h in 1:4
    pc_start = (h-1) * pcs_per_head_4 + 1
    pc_end   = h * pcs_per_head_4
    head_var = sum(F.S[pc_start:pc_end] .^ 2)
    println("Head $h (PCs $pc_start-$pc_end): $(round(100*head_var/total_var, digits=1))% variance")
end

# ── Save grids ───────────────────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)
for (label, samps, fname) in [
    ("H=1 β=200",  sa1_samples,     "Fig_multihead_h1_gen.png"),
    ("H=2 β=200",  mh_samples_gen[2], "Fig_multihead_h2_gen.png"),
    ("H=4 β=200",  mh_samples_gen[4], "Fig_multihead_h4_gen.png"),
    ("H=5 β=200",  mh_samples_gen[5], "Fig_multihead_h5_gen.png"),
    ("H=4 β=2000", sa4_ret,         "Fig_multihead_h4_ret.png"),
]
    save(joinpath(figpath, fname), Gray.(build_grid(samps)))
    @info "Saved $fname"
end

println("\nDone.")
flush(stdout); flush(stderr)
