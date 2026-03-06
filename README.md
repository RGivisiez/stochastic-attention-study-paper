# Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy

This repository contains the code, data, and manuscript source for the paper *Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy*. The central idea is to treat the energy function of a modern Hopfield network as a Boltzmann density and sample from it using the Unadjusted Langevin Algorithm (ULA). The resulting stochastic attention sampler interpolates continuously between exact pattern retrieval at high inverse temperature and free generation at low inverse temperature, producing novel outputs that are structured by the geometry of the stored memory matrix without any gradient-based training.

## What is in this repository

The repository is organized into two top-level directories. The `code/` directory holds all Julia source code, experiment notebooks, and pre-generated datasets. The `paper/` directory contains the LaTeX manuscript, bibliography, and style files needed to build the PDF.

### Code

All computational work lives under `code/`. The entry point for every notebook is `Include.jl`, which activates the Julia project environment, installs any missing dependencies from `Project.toml`, loads the required packages, and includes the three source files that define the core library.

The source modules in `code/src/` are:

- **`Compute.jl`** implements the stochastic attention sampler itself (Algorithm 1 in the paper). The `sample` function takes a memory matrix, an initial state, and a number of iterations, and returns the full trajectory of the Langevin chain. The update rule combines a contraction toward the origin, a softmax-weighted pull toward stored memories, and isotropic Gaussian noise scaled by the inverse temperature.

- **`Utilities.jl`** provides the diagnostic functions used throughout the experiments. These include `nearest_cosine_similarity`, which measures how close a state vector is to the nearest stored pattern; `hopfield_energy`, which evaluates the modern Hopfield energy at a given state using a numerically stable log-sum-exp computation; and `attention_entropy`, which computes the Shannon entropy of the softmax attention weights as an order parameter for the retrieval-to-generation transition.

- **`Data.jl`** contains `datagenerate`, which constructs synthetic memory matrices whose columns are independent draws from the uniform distribution on the unit sphere. It supports optional on-disk persistence via JLD2 archives and CSV files for portability, and accepts a random seed for exact reproducibility.

The experiment notebooks, each prefixed with `StochasticAttention-`, correspond to the sections of the paper:

- **`StochasticAttention-DataGeneration.ipynb`** generates all synthetic datasets used by the other notebooks and saves them to disk under `code/data/`. Running this notebook first ensures that every downstream experiment loads identical data regardless of platform or Julia version.

- **`StochasticAttention-Experiment-1-Temperature.ipynb`** sweeps the inverse temperature parameter across five orders of magnitude and records the cosine similarity to the nearest stored pattern and the entropy of the attention weights. The result is a dual-axis figure showing the smooth phase transition between the disordered (generative) and ordered (retrieval) regimes.

- **`StochasticAttention-Experiment-2-Convergence.ipynb`** runs multiple independent chains from different initializations on a small system where thorough mixing is feasible. It compares the pooled energy histogram from short chains to a long-run reference distribution and reports the Kolmogorov–Smirnov statistic and moment differences as a convergence diagnostic.

- **`StochasticAttention-Experiment-3-LoadRatio.ipynb`** constructs a phase diagram over the load ratio K/d and the inverse temperature, computing the attention concentration metric across a grid of conditions. The resulting heatmap reveals the boundary between the retrieval and diffuse regimes as a function of memory capacity and temperature.

- **`StochasticAttention-Experiment-5-Alternative-Market.ipynb`** applies the sampler to S&P 500 equity log-returns (d=424 tickers, K=2,766 trading days). It evaluates marginal fidelity via per-ticker KS tests (with per-ticker affine correction for variance compression), cross-asset Frobenius correlation error, novelty relative to historical scenarios, and temporal stylized facts (return autocorrelation, volatility clustering) via a MALA warm-start sequential protocol.

- **`StochasticAttention-Experiment-6-Simpsons.ipynb`** scales the sampler to K=100 Simpsons character face images at d=4,096. It demonstrates the SNR-based β-selection rule across a 5.2× dimension increase over MNIST and confirms that the relative ranking of methods (SA ≈ MALA ≫ all non-Langevin baselines) is preserved.

Pre-generated datasets live under `code/data/`, organized by experiment. Each experiment subdirectory contains a `data.jld2` archive (the canonical data source loaded by the notebooks) and individual `patterns_*.csv` files for inspection or use outside Julia.

### Standalone Julia scripts

Several experiments are also available as self-contained Julia scripts that can be run from the command line without Jupyter:

- **`code/mnist-experiment/run_multidigit_experiment.jl`** — runs the full 30-chain MNIST baseline comparison (SA, MALA, bootstrap, Gaussian perturbation, random convex combination, GMM-PCA, VAE) on digits 1, 3, and 8. Execute from `code/mnist-experiment/`:
  ```
  julia --project=. run_multidigit_experiment.jl
  ```

- **`code/mnist-experiment/run_single_chain_experiment.jl`** — runs a single long chain (T=50,000) on digit 3 at β∈{2000,200,50} to decompose multi-chain diversity into initialization and within-chain mixing contributions. Execute from `code/mnist-experiment/`:
  ```
  julia --project=. run_single_chain_experiment.jl
  ```

- **`code/mnist-experiment/run_gmm_experiment.jl`** — runs the GMM-PCA baseline (PCA to r=50 dimensions, 10-component diagonal-covariance GMM via EM, sample and reconstruct) on digit 3. Produces the GMM-PCA row in Table 1. Execute from `code/mnist-experiment/`:
  ```
  julia --project=. run_gmm_experiment.jl
  ```

- **`code/mnist-experiment/run_sa_beta200_digit3.jl`** — runs SA and MALA at β=200 (generation regime) on digit 3 using the identical 30-chain protocol. Produces the SA (β=200) row in Table 1. Execute from `code/mnist-experiment/`:
  ```
  julia --project=. run_sa_beta200_digit3.jl
  ```

- **`code/mnist-experiment/run_before_after.jl`** — generates a before/after figure for digit 8, showing stored patterns alongside generated samples from the same chains. Execute from `code/mnist-experiment/`:
  ```
  julia --project=. run_before_after.jl
  ```

- **`code/mnist-experiment/run_temp_spectrum_digit8.jl`** — generates temperature spectrum visualizations on digit 8, showing how samples change across β ∈ {10, 50, 200, 2000}. Execute from `code/mnist-experiment/`:
  ```
  julia --project=. run_temp_spectrum_digit8.jl
  ```

- **`code/vae-experiment/run_vae_experiment.jl`** — trains a small VAE (latent dim 8, two-phase training to prevent posterior collapse) on the same K=100 digit-3 patterns used in the main MNIST experiment, generates 150 samples, and reports novelty/diversity/energy metrics with standard errors. Results: N=0.214±0.005, D̄=0.441±0.008, Ē=−0.286±0.005 — the strongest non-Langevin baseline. Execute from `code/vae-experiment/`:
  ```
  julia --project=../mnist-experiment run_vae_experiment.jl 2>&1 | tee vae_results.log
  ```

### Paper

The manuscript source is in `paper/`. The main file is `Paper_v1.tex`, which inputs individual section files from `paper/sections/` (introduction, background, method, related work, experiments, discussion, appendix). References are in `References_v1.bib`. The `Build.sh` script runs the full `pdflatex → bibtex → pdflatex → pdflatex` cycle to produce the final PDF.

## Getting started

### Installing Julia

The code requires [Julia](https://julialang.org/downloads/) version 1.10 or later. On macOS the simplest route is Homebrew (`brew install julia`); on Linux and Windows, download the official binaries from the Julia website and follow the platform-specific instructions to add `julia` to your PATH. You can verify the installation by running `julia --version` in a terminal.

### Running the notebooks

The notebooks are standard Jupyter notebooks with a Julia kernel. If you do not already have Jupyter installed, Julia's `IJulia` package (listed in the project dependencies) will offer to install it for you the first time you run `using IJulia; notebook()` from the Julia REPL. Alternatively, you can open the notebooks directly in Visual Studio Code with the Julia and Jupyter extensions, which is the workflow used during development.

To run the code, clone the repository and navigate to the `code/` directory:

```
git clone https://github.com/varnerlab/stochastic-attention-model.git
cd stochastic-attention-model/code
```

Open any notebook and execute the first cell, which calls `include("Include.jl")`. On the first run this will activate the project environment defined by `Project.toml`, download and precompile all dependencies (Plots, StatsPlots, JLD2, NNlib, Distributions, and others), and load the source library. Subsequent runs will skip the installation step and start almost immediately. The pre-generated datasets in `code/data/` mean you can run the experiment notebooks directly without first running the data generation notebook, though regenerating the data from scratch is supported and will produce identical results thanks to fixed random seeds.

### Building the paper

Building the PDF requires a LaTeX distribution with `pdflatex` and `bibtex`. [TeX Live](https://tug.org/texlive/) (Linux/macOS) or [MiKTeX](https://miktex.org/) (Windows) both work. From the `paper/` directory:

```
./Build.sh Paper_v1
```

This produces `Paper_v1.pdf`. The build script is a thin wrapper around the standard four-pass compilation cycle and assumes the figure PDFs have already been generated by the experiment notebooks.

## License

This project is released under the [MIT License](LICENSE).
