include("common.jl")
include("wrappers.jl")

# Waveform example
function waveform()
    n_cv, n_test = 375, 10^3
    nsamples = n_cv + n_test
    nfeatures = 21
    L, X = MVDA.waveform(nsamples, nfeatures; rng=StableRNG(1903))
    return DataFrame([L X], :auto)
end

function BRCA()
  data = CSV.read("/home/alanderos/Desktop/BRCA/TCGA-BRCA-preprocessed.csv", DataFrame)
  X = select(data, Not(["patient", "barcode", "sample", "subtype"])) |> Matrix{Float64}
  @. X = log2(1 + X)
  L = data[!, "subtype"]
  return DataFrame([L X], :auto)
end

function save_svm_results(dir::String, title::String, algorithm, result::NT;
  overwrite::Bool=false,
) where NT
#
  dir = joinpath(dir, "modelA")
  if !ispath(dir)
      mkpath(dir)
  end

  # Filenames
  fit_filename = joinpath(dir, "summary.out")

  # Other Setttings/Parameters
  delim = ','
  alg = algorithm
  hyperparameters = result.hyperparameters
  cost = hyperparameters.cost

  # Fit Result
  fit_header = ("title", "algorithm", "replicate", "cost", "active_variables", "time", "train", "test",)
  replicate = MVDA.init_report(fit_filename, fit_header, delim, overwrite)
  open(fit_filename, "a") do io
      fit_data = (title, alg, replicate, cost,
          length(result.fit.support),
          result.fit.time,
          result.fit.train.score,
          result.fit.test.score,
      )
      write(io, join(fit_data, delim), '\n')
  end

  # Additional files nested within directory.
  rep = string(replicate)
  rep_dir = joinpath(dir, rep)
  if !ispath(rep_dir)
      mkpath(rep_dir)
  end
  mat_filename = joinpath(rep_dir, "confusion_matrix.out")

  # Confusion Matrix
  train_mat, d = result.fit.train.confusion_matrix
  test_mat, _ = result.fit.test.confusion_matrix
  labels, idx = collect(keys(d)), collect(values(d))
  labels .= labels[idx]
  open(mat_filename, "w") do io
      header = ("subset", "true/predicted", labels...)
      write(io, join(header, delim), '\n')
      for i in eachindex(labels)
          row_data = ("train", labels[i], train_mat[i, :]...)
          write(io, join(row_data, delim), '\n')
      end
      for i in eachindex(labels)
          row_data = ("test", labels[i], test_mat[i, :]...)
          write(io, join(row_data, delim), '\n')
      end
  end

  return nothing
end

function run_all_examples(dir, seed, examples)
  for (example, prob_settings, cv_settings, nhyper, data_transform, preshuffle) in examples
    df = if example == "waveform"
        waveform()
    elseif example == "BRCA"
        BRCA()
    else
        MVDA.dataset(example)
    end
    # Get the data
    input_data = (string.(df[!,1]), Matrix{Float64}(df[!,2:end]))

    # Shuffle data before splitting.
    rng = StableRNG(seed)
    if preshuffle
        data = getobs(shuffleobs(input_data, ObsDim.First(), rng), ObsDim.First())
    else
        data = getobs(input_data, ObsDim.First())
    end

    _, intercept = prob_settings
    (nfolds, nreplicates, split) = cv_settings
    grid = MVDA.make_log10_grid(-5, 5, nhyper)

    example_dir = joinpath(dir, example, "L1RSVM")
    if !ispath(example_dir)
      mkpath(example_dir)
      @info "Created directory for example $(example)" output_dir=example_dir
    end

    rng = StableRNG(seed)
    for i in 1:nreplicates
      @info "Repeated CV | Replicate $(i) / $(nreplicates)"
      shuffled = shuffleobs(data, obsdim=ObsDim.First(), rng=rng)

      result = L1R_L2LOSS_SVC(shuffled[1], shuffled[2];
        is_class_specific=false,
        data_transform=ZScoreTransform,
        at=split,
        nfolds=nfolds,
        Cvals=grid,
        bias=Float64(intercept-1),
        verbose=true,
        tolerance=1e-3,
        seed=i,
      )

      if i == 1
        save_svm_results(example_dir, example, "L1RSVM", result; overwrite=true)
      else
        save_svm_results(example_dir, example, "L1RSVM", result; overwrite=false)
      end
    end
  end
end

examples = (
  # ("iris", (nothing, true,), (3, 50, 120/150,), 25, ZScoreTransform, true),
  # ("lymphography", (nothing, false,), (3, 50, 105 / 148,), 37, NoTransformation, true),
  # ("zoo", (nothing, true,), (3, 50, 0.9,), 37, NoTransformation, true),
  # ("bcw", (nothing, true,), (5, 50, 0.8,), 37, NoTransformation, true),
  # ("waveform", (nothing, false,), (5, 50, 375 / 2575,), 37, ZScoreTransform, true),
  # ("splice", (nothing, false,), (5, 50, 0.685,), 109, NoTransformation, true),
  # ("letters", (nothing, true,), (5, 50, 0.8,), 37, NoTransformation, true),
  # # use cv / test split in original dataset: 3823 + 1797
  # ("optdigits", (nothing, true,), (5, 50, 3823 / 5620,), 61, NoTransformation, false),
  # # use cv / test split in original dataset: 528 + 462
  # ("vowel", (nothing, true,), (5, 50, 528 / 990,), 37, NoTransformation, false),
  # # use cv / test split in original dataset: 7352 + 2947
  #
  ("HAR", (nothing, true,), (5, 50, 7352 / 10299,), 109, NoTransformation, false),
  ("TCGA-HiSeq", (nothing, true,), (4, 10, 0.75,), 997, ZScoreTransform, true),
  # ("BRCA", (nothing, true,), (5, 10, 0.70,), 107, ZScoreTransform, true),
)

dir = ARGS[1]               # output directory
seed = parse(Int, ARGS[2])  # seed for StableRNG
@info "Running linear benchmarks" dir=dir seed=seed

run_all_examples(dir, seed, examples)
