#!/usr/bin/zsh

# Check that Julia command is available.
if [ -n "${JULIA+1}" ]; then
    alias julia="$JULIA/julia"
fi

export JULIA_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export OUTPUTDIR=/home/alanderos/Desktop/VDA-Results

echo "Running cancer data examples"
julia -t ${JULIA_NUM_THREADS} examples/cancer.jl ${OUTPUTDIR}/cancer

echo "Running examples with linear classifiers"
julia -t ${JULIA_NUM_THREADS} examples/linear.jl ${OUTPUTDIR}/linear

echo "Running examples with nonlinear classifiers"
julia -t ${JULIA_NUM_THREADS} examples/nonlinear.jl ${OUTPUTDIR}/nonlinear
