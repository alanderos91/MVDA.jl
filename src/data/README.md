# MVDA Examples

You can load an example by invoking `MVDA.dataset(name)`.
The list of available datasets is accessible via `MVDA.list_datasets()`.

Please note that the descriptions here are *very* brief summaries. Follow the links for additional information.

## Dataset: HAR

**6 classes / 10299 instances / 561 variables**

See: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

Check the README.txt file for further details about this dataset.

**Notes**: 

- Features are normalized and bounded within [-1,1].
- Each feature vector is a row on the text file.
- The units used for the accelerations (total and body) are 'g's (gravity of earth -> 9.80665 m/seg2).
- The gyroscope units are rad/seg.
- A video of the experiment including an example of the 6 recorded activities with one of the participants can be seen in the following link: http://www.youtube.com/watch?v=XOEN9W05_4A

For more information about this dataset please contact: activityrecognition '@' smartlab.ws

**License**:

Use of this dataset in publications must be acknowledged by referencing the following publication [1] 

[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. 

This dataset is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.

## Dataset: TCGA-HiSeq

**5 classes / 801 instances / 20265 variables**

See: https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

Original labels for genes can be found at https://www.synapse.org/#!Synapse:syn4301332.
The relevant file is unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp, which has genes along rows and
instance along columns.

Extracting the correct gene labels requires cross-referencing Tissue Source Site (TSS) and Study codes.

- TSS: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes
- Study: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations

267 genes are dropped due to having zero expression through all samples.

## Dataset: bcw (Breast Cancer Wisconsin)

**2 classes / 683 instances (16 dropped) / 9 variables**

See: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

16 instances from the original dataset are dropped due to missing values.

## Dataset: iris

**3 classes / 150 instances / 4 variables**

See: https://archive.ics.uci.edu/ml/datasets/iris

## Dataset: letters

**26 classes / 20000 instances / 16 variables**

See: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

## Dataset: lymphography

**4 classes / 148 instances / 18 variables**

See: https://archive.ics.uci.edu/ml/datasets/Lymphography

The original 18 features have been expanded to 

## Dataset: optdigits

**10 classes / 5620 instances / 64 variables**

See: https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits

## Dataset: spiral

**3 classes / 1000 instances / 2 variables**

Adapted from: https://smorbieu.gitlab.io/generate-datasets-to-understand-some-clustering-algorithms-behavior/

Simulation of a noisy pattern with 3 spirals.

## Dataset: spiral-hard

**3 classes / 1000 instances / 2 variables**

Adapted from: https://smorbieu.gitlab.io/generate-datasets-to-understand-some-clustering-algorithms-behavior/

Simulation of a noisy pattern with 3 spirals.

Bayes error is expected to be ≈0.1 due to random class inversions.

## Dataset: splice

**3 classes / 3176 instances (14 dropped) / 180 variables**

See: https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)

The original sequence of 60 nucleotides is expanded to 180 variables using the binary encoding

    T ==> [0,0,0]
    G ==> [0,0,1]
    C ==> [0,1,0]
    A ==> [1,0,0]

14 instances with ambiguous sequences are dropped.

## Dataset: synthetic

**2 classes / 1000 instances / 500 variables**

A simulated multivariate normal.

Classes are determined by the first two variables using the signs of `X*b`
using `b[1] = 10`, `b[2] = -10`, and `b[j] = 0` otherwise.

Covariance structure is as follows

```julia
Σ[1,1] = 1
Σ[2,2] = 1
Σ[i,i] = 1e-2 for i=3, 4, …, 500
Σ[1,2] = 0.4
Σ[i,j] = 1e-4
```

## Dataset: synthetic-hard

**2 classes / 1000 instances / 500 variables**

A simulated multivariate normal.

Classes are determined by the first two variables using the signs of `X*b`
using `b[1] = 10`, `b[2] = -10`, and `b[j] = 0` otherwise.

Covariance structure is as follows

```julia
Σ[1,1] = 1
Σ[2,2] = 1
Σ[i,i] = 1e-2 for i=3, 4, …, 500
Σ[1,2] = 0.4
Σ[i,j] = 1e-4
```

Bayes error is expected to be ≈0.1 due to random class inversions.

## Dataset: vowel

**11 classes / 990 instances / 10 variables**

See: https://web.stanford.edu/~hastie/ElemStatLearn/ and https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.info.txt

## Dataset: zoo

**7 classes / 101 instances / 16 variables**

See: https://archive.ics.uci.edu/ml/datasets/zoo

*Note*: This version strips 'animal name' from attributes.
 
