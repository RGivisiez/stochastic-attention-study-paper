#!/usr/bin/env julia
# Quick hyperparameter sweep for VAE baseline.
# Run from repo root: julia code/vae-experiment/sweep_vae.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "mnist-experiment"))
include(joinpath(@__DIR__, "..", "mnist-experiment", "Include-MNIST.jl"))
using Flux, Statistics, Random, LinearAlgebra, Printf

const DIGIT = 3; const K = 100; const D = 784
const β_inv_temp = 2000.0; const S = 150; const NC = 30; const SPC = 5

@info "Loading MNIST patterns …"
dict = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K)
X_raw = zeros(Float64, D, K)
for i in 1:K
    img = dict[DIGIT][:, :, i]
    X_raw[:, i] = reshape(transpose(img) |> Matrix, D) |> vec
end
X̂ = mapslices(v -> v ./ (norm(v) + 1e-12), X_raw; dims=1)
X_train = Float32.(X̂)

# ── Model struct (defined at module scope for Flux.@layer) ────────────────────
struct SweepVAE
    enc   # shared trunk
    enc_μ
    enc_lσ
    dec
end
Flux.@layer SweepVAE

function build_vae(latent::Int)
    SweepVAE(
        Chain(Dense(D => 256, relu), Dense(256 => 128, relu)),
        Dense(128 => latent),
        Dense(128 => latent),
        Chain(Dense(latent => 128, relu), Dense(128 => 256, relu), Dense(256 => D))
    )
end

function fwd(m::SweepVAE, x, ε)
    h  = m.enc(x)
    μ  = m.enc_μ(h);  lσ = m.enc_lσ(h)
    z  = μ .+ exp.(0.5f0 .* lσ) .* ε
    o  = m.dec(z)
    x̂  = o ./ (sqrt.(sum(o.^2; dims=1)) .+ 1f-8)
    return x̂, μ, lσ
end

function chain_se(vals, f)
    std([f(vals[(i-1)*SPC+1:i*SPC]) for i in 1:NC]) / sqrt(NC)
end

# ── Run one config ────────────────────────────────────────────────────────────
function run_config(latent, β_kl, phase1, phase2; seed=42)
    Random.seed!(seed)
    m = build_vae(latent)
    opt = Flux.setup(Adam(1f-3), m)

    # Phase 1: reconstruction only (force decoder to use z)
    for ep in 1:phase1
        ε = randn(Float32, latent, K)
        _, grads = Flux.withgradient(m) do md
            x̂, _, _ = fwd(md, X_train, ε)
            mean(sum((X_train .- x̂).^2; dims=1))
        end
        Flux.update!(opt, m, grads[1])
    end

    # Phase 2: β-VAE with linear KL warmup
    for ep in 1:phase2
        kl_w = Float32(β_kl) * Float32(ep) / Float32(phase2)
        ε = randn(Float32, latent, K)
        _, grads = Flux.withgradient(m) do md
            x̂, μ, lσ = fwd(md, X_train, ε)
            recon = mean(sum((X_train .- x̂).^2; dims=1))
            kl    = -0.5f0 * mean(sum(1f0 .+ lσ .- μ.^2 .- exp.(lσ); dims=1))
            recon + kl_w * kl
        end
        Flux.update!(opt, m, grads[1])
    end

    # Generate 150 samples
    Random.seed!(9999)
    Z = randn(Float32, latent, S)
    o = m.dec(Z)
    raw = o ./ (sqrt.(sum(o.^2; dims=1)) .+ 1f-8)
    samps = [Float64.(raw[:, i]) for i in 1:S]

    N_v = [sample_novelty(s, X̂)            for s in samps]
    D_v = sample_diversity(samps)
    E_v = [hopfield_energy(s, X̂, β_inv_temp) for s in samps]

    N_se = chain_se(N_v, mean)
    D_se = std([sample_diversity(samps[(i-1)*SPC+1:i*SPC]) for i in 1:NC]) / sqrt(NC)
    E_se = chain_se(E_v, mean)

    return mean(N_v), N_se, D_v, D_se, mean(E_v), E_se
end

# ── Sweep configs ─────────────────────────────────────────────────────────────
configs = [
    (8,  0.010f0, 2000, 2000, "lat=8  β=0.010 p1=2k p2=2k"),
    (8,  0.001f0, 2000, 2000, "lat=8  β=0.001 p1=2k p2=2k"),
    (8,  0.010f0, 3000, 2000, "lat=8  β=0.010 p1=3k p2=2k"),
    (16, 0.010f0, 2000, 2000, "lat=16 β=0.010 p1=2k p2=2k"),
    (16, 0.001f0, 2000, 2000, "lat=16 β=0.001 p1=2k p2=2k"),
]

println("\n$(rpad("Config", 36)) │  N̄ ± SE     │  D̄ ± SE     │  Ē ± SE")
println("─"^82)
for (lat, bkl, p1, p2, lbl) in configs
    @info "Running: $lbl"
    N, Nse, D, Dse, E, Ese = run_config(lat, bkl, p1, p2)
    @printf("%-36s │ %.3f ± %.3f │ %.3f ± %.3f │ %.3f ± %.3f\n",
            lbl, N, Nse, D, Dse, E, Ese)
end
println("─"^82)
println("$(rpad("GMM-PCA (reference)", 36)) │ 0.198 ± 0.004 │ 0.419 ± 0.011 │ -0.303 ± 0.005")
println("$(rpad("VAE lat=32 β=0.01 (current)", 36)) │ 0.103 ± 0.004 │ 0.333 ± 0.011 │ -0.397 ± 0.004")
