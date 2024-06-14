# Examples & Reproducibility

Output of Julia's `versioninfo()`:

```text
Julia Version 1.7.3
Commit 742b9abb4d (2022-05-06 12:58 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, skylake)
```

To reproduce our work you will need to `instantiate()` the dependencies used in the `examples/` directory. In a fresh Julia session, run:

```julia
#
# Verify you are in the `examples/` directory, otherwise
# use cd() to navigate to it.
#
pwd()

#
# Load Julia's package manager, activate the environment in
# the `examples/` directory, and instantiate.
#
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

This will take a while but is only run once. In the future you can simply do `import Pkg; Pkg.activate(".")` as is done in the scripts.

## Installing Packages for Benchmarks

The benchmarks require a few external packages. We use Julia's package manager and Conda to handle these dependencies.

### R Package: MGSDA

**You will need to run this on your machine. Here we use Julia's Conda interface to assist.**

```julia
import Pkg, Conda
#
# High-level settings
# 
rpkgs = ["r-mgsda"] # packages to install
rmvda = :MVDA       # environment to keep things separate
channel = "r"
#
# Install separate R managed by Conda
#
Conda.add("r-base", rmvda, channel=channel)
#
# Install R packages used in our code
#
for rpkg in rpkgs
  Conda.add(rpkg, rmvda, channel=channel)
end
#
# Setup RCall to use our R installation
#
ENV["R_HOME"] = joinpath(Conda.lib_dir(rmvda), "R")
Pkg.build("RCall")
#
# Test that everything works correctly
#
using RCall

R"""
sessionInfo()
"""

R"""
rinfo <- as.data.frame(installed.packages())
subset(rinfo,
  Package %in% c("MGSDA"),
  select=c(Version, Built, LibPath)
)
"""
```

Sample output for R's `sessioninfo()`

```text
R version 4.3.1 (2023-06-16)
Platform: x86_64-conda-linux-gnu (64-bit)
Running under: Manjaro Linux

Matrix products: default
BLAS/LAPACK: /home/alanderos/.julia/conda/3/envs/MVDA/lib/libopenblasp-r0.3.21.so;  LAPACK version 3.9.0

locale:
 [1] LC_CTYPE=en_US.utf8           LC_NUMERIC=C                 
 [3] LC_TIME=en_US.UTF-8           LC_COLLATE=en_US.utf8        
 [5] LC_MONETARY=en_US.UTF-8       LC_MESSAGES=en_US.utf8       
 [7] LC_PAPER=en_US.UTF-8          LC_NAME=en_US.UTF-8          
 [9] LC_ADDRESS=en_US.UTF-8        LC_TELEPHONE=en_US.UTF-8     
[11] LC_MEASUREMENT=en_US.UTF-8    LC_IDENTIFICATION=en_US.UTF-8

time zone: America/Los_Angeles
tzcode source: system (glibc)

attached base packages:
[1] stats     graphics  grDevices utils     datasets  methods   base     

loaded via a namespace (and not attached):
[1] compiler_4.3.1
```

Sample output for `rinfo`:

```text
      Version Built                                                LibPath
MGSDA   1.6.1 4.3.1 /home/alanderos/.julia/conda/3/envs/MVDA/lib/R/library
```

### Julia Wrappers: LIBLINEAR via LIBSVM.jl

**You do not need to run this! It is only provided to record how LIBSVM was installed**.

```julia
import Pkg
Pkg.add("LIBSVM", rev="1ce6283")
```

## Running the Examples

We provide a script `run.sh` to run *all* the benchmarks. **It will take a long time to run!**.

### Synthetic Benchmarks

The script `synth_homogeneous.jl` compares VDA classifiers against MGSDA and the **linear** $\ell_{1}$-regularized $\ell_{2}$-SVM from LIBLINEAR, accessed via LIBSVM.jl (this means the primal problem is solved). This benchmark simulates 3-class and 10-class data with 1000 multivariate Gaussian features under a Toeplitz correlation structure. **Each class is associated with the same set of $k=30$ features; they are *homogeneous classes***. The number of samples, signal-to-noise ratio, and correlation strength are varied.

The script `synth_heterogeneous.jl` compares VDA classifiers using *homogeneous* and *heterogeneous* variable selection. The simulation setup is identical to `synth_homogeneous.jl` except now each of the $c=3$ or $c=10$ classes may have a different set of $k=10$ or $k=3$ features associated with it, respectively. **They are heterogeneous classes in this sense**.

### Cancer Benchmarks

### Linear Benchmarks (UCI Machine Learning Repository)

### Nonlinear Benchmarks
