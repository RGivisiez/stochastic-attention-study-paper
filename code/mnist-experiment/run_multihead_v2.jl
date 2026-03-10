#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Multi-Head SA Demo v2 (Q5 response)
#
# Two questions:
#   A. Does multi-head SA improve sample quality at fixed K?
#   B. Does increasing K achieve the same effect without multi-head?
#
# Configurations (all at β=200, generation regime):
#   1. Single-head, K=100
#   2. Multi-head H=4 (25 PCs/head), K=100
#   3. Single-head, K=500
#   4. Single-head, K=1000
#   5. Multi-head H=4, K=500
#   6. Multi-head H=4, K=1000
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO, Printf
flush(stdout); flush(stderr)
@info "Environment loaded."

# ── Parameters ───────────────────────────────────────────────────────────────
const number_of_rows      = 28
const number_of_cols      = 28
const number_of_pixels    = number_of_rows * number_of_cols
const α_step              = 0.01
const β_gen               = 200.0
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
    H, W = 28, 28
    canvas = zeros(Float64, nrows*H + (nrows-1)*gap, ncols*W + (ncols-1)*gap)
    indices = round.(Int, range(1, length(samples), length=nrows*ncols))
    for idx in 1:(nrows*ncols)
        r = div(idx-1, ncols); c = rem(idx-1, ncols)
        y0 = r*(H+gap)+1; x0 = c*(W+gap)+1
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

ϵ_norm = 1e-12

# ── Load MNIST at multiple K values ──────────────────────────────────────────
K_values = [100, 500, 1000]
memories = Dict{Int, Matrix{Float64}}()

for K_target in K_values
    @info "Loading K=$K_target digit-$DIGIT patterns …"
    flush(stdout)
    digits_dict = MyMNISTHandwrittenDigitImageDataset(number_of_examples=K_target)

    n_avail = size(digits_dict[DIGIT], 3)
    K_actual = min(K_target, n_avail)

    X̂ = zeros(Float64, number_of_pixels, K_actual)
    for i in 1:K_actual
        xᵢ = reshape(transpose(digits_dict[DIGIT][:, :, i]) |> Matrix, number_of_pixels) |> vec
        X̂[:, i] = xᵢ ./ (norm(xᵢ) + ϵ_norm)
    end
    memories[K_actual] = X̂
    @info "  Loaded: $(size(X̂)) (K=$K_actual)"
    flush(stdout)
end

# ── Single-head SA runner ────────────────────────────────────────────────────
function run_single_head(X̂; β=200.0)
    K = size(X̂, 2)
    d = size(X̂, 1)
    samples = Vector{Vector{Float64}}()
    Random.seed!(42)
    pidx = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
    for (c, k) in enumerate(pidx)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(d)
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=β, α=α_step, seed=12345 + c)
        pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
        na = length(pool)
        idxs = round.(Int, range(1, na, length=min(samples_per_chain, na)))
        for idx in idxs; push!(samples, pool[idx]); end
    end
    return samples
end

# ── Multi-head PCA SA runner ─────────────────────────────────────────────────
function run_multihead(X̂, n_heads; β=200.0)
    d, K = size(X̂)

    # PCA
    X_mean = mean(X̂, dims=2) |> vec
    X_centered = X̂ .- X_mean
    F = svd(X_centered)
    n_pcs = min(K, size(F.U, 2))
    # Make n_pcs divisible by n_heads
    n_pcs = (n_pcs ÷ n_heads) * n_heads
    PC = F.U[:, 1:n_pcs]
    pcs_per_head = n_pcs ÷ n_heads

    samples = Vector{Vector{Float64}}()
    Random.seed!(42)
    pidx = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))

    for (c, k) in enumerate(pidx)
        seed_c = 12345 + c
        head_traj = Vector{Matrix{Float64}}(undef, n_heads)

        for h in 1:n_heads
            pc_s = (h-1)*pcs_per_head + 1
            pc_e = h*pcs_per_head
            W_h = PC[:, pc_s:pc_e]'

            M_h = W_h * X̂
            for ki in 1:K
                nrm = norm(M_h[:, ki])
                nrm > ϵ_norm && (M_h[:, ki] ./= nrm)
            end

            Random.seed!(seed_c)
            z₀ = W_h * (X̂[:, k] .+ σ_init .* randn(d))
            (_, Ξ_h) = sample(M_h, z₀, T_per_chain; β=β, α=α_step, seed=seed_c + h)
            head_traj[h] = Ξ_h
        end

        pool = Vector{Vector{Float64}}()
        for tᵢ in (T_burnin+1):thin_interval:T_per_chain
            ξ = copy(X_mean)
            for h in 1:n_heads
                pc_s = (h-1)*pcs_per_head + 1
                pc_e = h*pcs_per_head
                ξ .+= PC[:, pc_s:pc_e] * head_traj[h][tᵢ, :]
            end
            push!(pool, ξ)
        end

        na = length(pool)
        idxs = round.(Int, range(1, na, length=min(samples_per_chain, na)))
        for idx in idxs; push!(samples, pool[idx]); end
    end

    return samples
end

# ══════════════════════════════════════════════════════════════════════════════
# Run all configurations
# ══════════════════════════════════════════════════════════════════════════════

results = Vector{Tuple{String, Int, Vector{Vector{Float64}}}}()

# Single-head at each K
for K_val in sort(collect(keys(memories)))
    X̂ = memories[K_val]
    K = size(X̂, 2)
    @info "Single-head SA, K=$K, β=$β_gen …"
    flush(stdout)
    samps = run_single_head(X̂; β=β_gen)
    push!(results, ("H=1", K, samps))
    @info "  Done: $(length(samps)) samples"
    flush(stdout)
end

# Multi-head (H=4) at each K
for K_val in sort(collect(keys(memories)))
    X̂ = memories[K_val]
    K = size(X̂, 2)
    @info "Multi-head SA (H=4), K=$K, β=$β_gen …"
    flush(stdout)
    samps = run_multihead(X̂, 4; β=β_gen)
    push!(results, ("H=4", K, samps))
    @info "  Done: $(length(samps)) samples"
    flush(stdout)
end

# ══════════════════════════════════════════════════════════════════════════════
# Compute metrics (always evaluated against the SAME memory used for sampling)
# ══════════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("MULTI-HEAD SA vs LARGER K — RESULTS")
println("="^80)
println("All at β=$β_gen (generation regime), digit-$DIGIT")
println("30 chains × 5000 steps, 2000 burn-in, thin/100, 5/chain = 150 total")
println("="^80)

println("\n%----- Table -----")
println("Config & K & Novelty & Diversity & Max-Cos \\\\")
println("\\midrule")

for (head_label, K_val, samps) in results
    X̂_ref = memories[K_val]

    nov_vals = [sample_novelty(ξ, X̂_ref) for ξ in samps]
    nov_mean = mean(nov_vals)
    nov_se   = chain_metric_se(samps, g -> mean(sample_novelty(ξ, X̂_ref) for ξ in g))

    div_mean = sample_diversity(samps)
    div_se   = chain_metric_se(samps, g -> sample_diversity(g))

    maxcos_vals = [1.0 - sample_novelty(ξ, X̂_ref) for ξ in samps]
    maxcos_mean = mean(maxcos_vals)
    maxcos_se   = chain_metric_se(samps, g -> mean(1.0 - sample_novelty(ξ, X̂_ref) for ξ in g))

    en_vals = [hopfield_energy(ξ, X̂_ref, β_gen) for ξ in samps]
    en_mean = mean(en_vals)

    println("$head_label & $K_val & " *
            "$(round(nov_mean, digits=3)) ± $(round(nov_se, digits=3)) & " *
            "$(round(div_mean, digits=3)) ± $(round(div_se, digits=3)) & " *
            "$(round(maxcos_mean, digits=3)) ± $(round(maxcos_se, digits=3)) \\\\")
end

# ── Save grids ───────────────────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)
for (head_label, K_val, samps) in results
    fname = "Fig_mhv2_$(head_label)_K$(K_val).png"
    save(joinpath(figpath, fname), Gray.(build_grid(samps)))
    @info "Saved $fname"
end

println("\nDone.")
flush(stdout); flush(stderr)
