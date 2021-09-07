# Multicategory Vertex Discriminant Analysis (MVDA)

A Julia package for classification using a vertex-valued encoding of data labels.
Work in progress.

### Loading demo datasets

```julia
using MVDA

MVDA.list_datasets() # lists available demo datasets
df = MVDA.dataset("synthetic") # loads the `synthetic` dataset as a DataFrame
```
