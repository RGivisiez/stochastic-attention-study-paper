#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# SA + MALA at β=200 for MNIST digit "3"
# Identical 30-chain protocol to run_multidigit_experiment.jl (β=2000) so
# results can be added as a new row in tab:mnist-baselines.
# Energy is evaluated at the sampling β (200), which is the natural
# thermodynamic quantity for the β=200 Boltzmann distribution.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO
@info "Environment loaded."

# ── Parameters (identical to run_multidigit_experiment.jl except β) ──────────
const number_of_examples  = 100
const number_of_rows      = 28
const number_of_cols      = 28
const number_of_pixels    = number_of_rows * number_of_cols
const α_step              = 0.01
const β_200               = 200.0      # ← new operating point
const S                   = 150        # total samples
const n_chains            = 30
const T_per_chain         = 5000
const T_burnin            = 2000
const thin_interval       = 100
const samples_per_chain   = 5
const σ_init              = 0.01
const DIGIT               = 3

# ── decode / grid helpers ─────────────────────────────────────────────────────
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

# ── chain_metric_se helper ───────────────────────────────────────────────────
function chain_metric_se(samps, metric_fn; nc=n_chains, spc=samples_per_chain)
    vals = [metric_fn(samps[(i-1)*spc+1:i*spc]) for i in 1:nc]
    return std(vals) / sqrt(nc)
end

# ── Load MNIST ────────────────────────────────────────────────────────────────
@info "Loading MNIST …"
digits_dict = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)
@info "MNIST loaded."

# ── Build normalised memory matrix ───────────────────────────────────────────
ϵ = 1e-12
X  = zeros(Float64, number_of_pixels, number_of_examples)
X̂  = zeros(Float64, number_of_pixels, number_of_examples)
for i in 1:number_of_examples
    xᵢ = reshape(transpose(digits_dict[DIGIT][:, :, i]) |> Matrix, number_of_pixels) |> vec
    X[:, i] = xᵢ
end
for i in 1:number_of_examples
    X̂[:, i] = X[:, i] ./ (norm(X[:, i]) + ϵ)
end
K = size(X̂, 2)
@info "Memory matrix: $(size(X̂))  (K=$K)"

# ── SA multi-chain at β=200 ───────────────────────────────────────────────────
@info "Running SA 30-chain at β=$β_200 …"
sa_samples = Vector{Vector{Float64}}()
Random.seed!(42)
pattern_indices = StatsBase.sample(1:K, n_chains, replace = (n_chains > K))
for (c, k) in enumerate(pattern_indices)
    Random.seed!(12345 + c)
    sₒ = X̂[:, k] .+ σ_init .* randn(number_of_pixels)
    (_, Ξ) = sample(X̂, sₒ, T_per_chain; β = β_200, α = α_step, seed = 12345 + c)
    chain_pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
    n_avail = length(chain_pool)
    idxs = round.(Int, range(1, n_avail, length = min(samples_per_chain, n_avail)))
    for idx in idxs; push!(sa_samples, chain_pool[idx]); end
end
@info "SA: $(length(sa_samples)) samples"

# ── MALA multi-chain at β=200 ─────────────────────────────────────────────────
@info "Running MALA 30-chain at β=$β_200 …"
mala_samples     = Vector{Vector{Float64}}()
mala_accept_rates = Float64[]
Random.seed!(42)
pattern_indices = StatsBase.sample(1:K, n_chains, replace = (n_chains > K))
for (c, k) in enumerate(pattern_indices)
    Random.seed!(12345 + c)
    sₒ = X̂[:, k] .+ σ_init .* randn(number_of_pixels)
    (_, Ξ, ar) = mala_sample(X̂, sₒ, T_per_chain; β = β_200, α = α_step, seed = 12345 + c)
    push!(mala_accept_rates, ar)
    chain_pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
    n_avail = length(chain_pool)
    idxs = round.(Int, range(1, n_avail, length = min(samples_per_chain, n_avail)))
    for idx in idxs; push!(mala_samples, chain_pool[idx]); end
end
mala_mean_ar = round(mean(mala_accept_rates), digits=4)
@info "MALA: $(length(mala_samples)) samples, acceptance rate = $mala_mean_ar"

# ── Compute metrics ───────────────────────────────────────────────────────────
methods = ["SA (β=200)"   => sa_samples,
           "MALA (β=200)" => mala_samples]

fmt(x, d)          = let v = round(x; digits=d); abs(v) == 0.0 ? abs(v) : v end
fmtpm(v, se, dv, ds) = "\$$(fmt(v,dv)) \\pm $(fmt(se,ds))\$"

println("\n% ─── β=200 digit-3 table rows (energy evaluated at β=200) ───")
for (name, samps) in methods
    nov_vals  = [sample_novelty(ξ, X̂) for ξ in samps]
    en_vals   = [hopfield_energy(ξ, X̂, β_200) for ξ in samps]   # β=200
    nov_mean  = mean(nov_vals)
    div_mean  = sample_diversity(samps)
    en_mean   = mean(en_vals)
    nov_se    = chain_metric_se(samps, g -> mean(sample_novelty(ξ, X̂) for ξ in g))
    div_se    = chain_metric_se(samps, g -> sample_diversity(g))
    en_se     = chain_metric_se(samps, g -> mean(hopfield_energy(ξ, X̂, β_200) for ξ in g))
    println("$(name) & $(fmtpm(nov_mean, nov_se, 3, 3)) & " *
            "$(fmtpm(div_mean, div_se, 3, 3)) & " *
            "$(fmtpm(en_mean, en_se, 3, 3)) \\\\")
    @info "$(name): N=$(round(nov_mean,digits=3)) D=$(round(div_mean,digits=3)) E=$(round(en_mean,digits=3))"
end
println("% MALA acceptance rate at β=200: $mala_mean_ar")

# ── Save SA grid PNG ──────────────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)
fname = "Fig_mnist_grid_sa_beta200_digit3.png"
save(joinpath(figpath, fname), Gray.(build_grid(sa_samples)))
@info "Saved $fname"
println("\nDone.")
