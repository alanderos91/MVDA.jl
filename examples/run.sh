#!/usr/bin/zsh

# Check that Julia command is available.
if [ -n "${JULIA+1}" ]; then
    alias julia="$JULIA/julia"
fi

export JULIA_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=1

echo "Running examples with linear classifiers"
julia -t ${JULIA_NUM_THREADS} --project=@. examples/linear.jl /home/alanderos/Desktop/VDA-Results/linear

echo "Running cancer data examples"
julia -t ${JULIA_NUM_THREADS} --project=@. examples/cancer.jl /home/alanderos/Desktop/VDA-Results/cancer

echo "Running examples with nonlinear classifiers"
julia -t ${JULIA_NUM_THREADS} --project=@. examples/nonlinear.jl /home/alanderos/Desktop/VDA-Results/nonlinear

echo "Running MS example"
julia -t ${JULIA_NUM_THREADS} --project=@. examples/ms.jl /home/alanderos/Desktop/VDA-Results/MS
