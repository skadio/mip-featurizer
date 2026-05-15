# mip-featurizer

Lightweight static featurizer for mixed-integer programming (MIP) instances.
It computes the 100 static instance-level features described in:

Hutter, F., Xu, L., Hoos, H. H., and Leyton-Brown, K. (2014).
*Algorithm runtime prediction: Methods & evaluation*.

## Requirements

- Python 3.9+
- A working Gurobi installation and license

Install Python dependencies:

```bash
python -m pip install -r requirement.txt
```

## Run Example

Run on a single MIP file:

```bash
python static_mip_feature_embedder.py path/to/instance.mps --output_json features.json
```

Run on multiple files and/or directories:

```bash
python static_mip_feature_embedder.py path/to/a.mps path/to/folder_with_mips
```

If `--output_json` is omitted, JSON is printed to stdout.

## Python API Example

```python
from static_mip_feature_embedder import StaticMIPFeatureEmbedder

embedder = StaticMIPFeatureEmbedder()
feature_dict = embedder.mip_file_to_feature_dict("path/to/instance.mps")

print(len(feature_dict))  # 100
print(feature_dict["n_vars"])
```