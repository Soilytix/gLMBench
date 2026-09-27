# Installing gLMBench

**Audience:** you have cloned gLMBench and want to score a model with it: the LOAM releases,
one of the external models the paper compares against, or your own.

**What the clone gives you:** the code, the model specs, the task manifests (every task file
is pinned by sha256), and the paper's reference numbers under `paper/`. It does **not** contain
model weights or task data. Both are fetched: weights from the Hugging Face Hub on first use,
task data by the scripts in `scripts/`.

---

## 0. TL;DR

```bash
git clone https://github.com/Soilytix/gLMBench.git && cd gLMBench
conda env create -f envs/glmbench.yml && conda activate glmbench
pip install -e ".[hf,data,dev,notebook]" -c envs/constraints-paper.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

pytest tests/unit                                                       # CPU only
glmbench run --model specs/echo-full.yaml --benchmark dummy_bench --results results/

hf auth login                                                           # gated repos: LOAM, NTv3
python scripts/fetch_bacbench_essentiality.py
python scripts/fetch_dgeb_ec_dna.py
python scripts/fetch_rnagym_prok.py
glmbench run --model specs/loam/LOAM-25M.yaml --benchmark loam_paper --results results/
```

Then [REPRODUCING_THE_PAPER.md](REPRODUCING_THE_PAPER.md) for the full set of models, and
[HOW_MODELS_ARE_SERVED.md](HOW_MODELS_ARE_SERVED.md) for what each model needs.

---

## 1. Environments

The benchmark core is torch-free; each model runs in a **runner** subprocess started with a
Python you choose (the spec's `runner.python_exe`). The paper used two environments:

| Env | File | Serves | Key versions |
| --- | --- | --- | --- |
| `glmbench` | `envs/glmbench.yml` | the core, and every model except Evo2: LOAM, Evo 1.5, GenomeOcean, gLM2, ProkBERT, NTv3 | Python 3.11, torch 2.12.0, transformers 5.12.1, einops, accelerate, safetensors, huggingface_hub; flash-attn 2.8.3 optional (Evo 1.5 only) |
| `evo2` | `envs/evo2.yml` | Evo2-7B only | Python 3.11, evo2 0.6.0, vtx 1.1.0, flash-attn 2.8.3; `glmbench` installed with `--no-deps` |

`envs/constraints-paper.txt` pins the exact core versions (numpy, scipy, scikit-learn, …) the
paper's probes ran on. The essentiality probe does not always converge, so its last digits can
depend on the scikit-learn/BLAS build: install against the constraints when you want the
paper's numbers exactly.

### The `glmbench` env

```bash
conda env create -f envs/glmbench.yml
conda activate glmbench
pip install -e ".[hf,data,dev,notebook]" -c envs/constraints-paper.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

The extras: `hf` (torch, transformers, safetensors, accelerate, einops; everything the Hugging
Face runners need), `data` (`datasets`, for the one-time task-data fetch), `dev` (pytest) and
`notebook` (Jupyter and matplotlib, for `notebooks/loam_hf_benchmark.ipynb`).
The core alone (`pip install -e .`) is enough to run the unit tests that need no model and to
render leaderboards.

Two things to get right:

- **The torch CUDA build must match your driver.** The pinned environment uses torch 2.12.0
  built for CUDA 12.6 (`cu126`). Check `nvidia-smi` and install the torch wheel for your driver
  if the default does not import with CUDA available.
- **transformers 5.12.1 is the tested version.** ProkBERT's and NTv3's remote modeling code was
  validated against it. Do **not** `pip install prokbert` or `pip install evo-model`: both pin
  an older transformers (and `evo-model` an older flash-attn) and will downgrade the env. The
  benchmark loads both models through their Hub remote code and needs neither package.

### flash-attn (only for Evo 1.5)

Evo 1.5's attention layers call flash-attn with no fallback, so it must import in the
`glmbench` env if you run that model. Nothing else in the env uses it, and LOAM does not need it.

Try the prebuilt wheel first:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
python -c "import flash_attn"
```

If the import fails (for example because the prebuilt wheel needs a newer system C library
than yours), or if no prebuilt wheel matches your torch (pip then compiles for every GPU
architecture, which takes hours), build from source, with a CUDA toolkit that matches your torch
build, restricted to your GPU's architecture so it takes minutes:

```bash
conda install -y -c nvidia cuda-nvcc=12.6 cuda-cudart-dev=12.6 cuda-cccl=12.6 cuda-nvtx-dev=12.6
pip install ninja packaging wheel
CUDA_HOME=$CONDA_PREFIX FLASH_ATTENTION_FORCE_BUILD=TRUE \
  FLASH_ATTN_CUDA_ARCHS=<XY> TORCH_CUDA_ARCH_LIST=<X.Y> MAX_JOBS=16 \
  pip install --no-build-isolation --no-binary :all: flash-attn==2.8.3
```

`<X.Y>` is your GPU's compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`),
and `<XY>` the same without the dot. The toolkit version (`12.6` above) should match the CUDA
version your torch was built for.

### The `evo2` env (only for Evo2-7B)

```bash
conda env create -f envs/evo2.yml
conda activate evo2
pip install torch==2.12.0 --extra-index-url https://download.pytorch.org/whl/cu126
pip install ninja packaging wheel
pip install --no-build-isolation flash-attn==2.8.3     # or the source build above
pip install evo2==0.6.0 vtx==1.1.0 numpy==2.4.6 pydantic==2.13.4 PyYAML==6.0.3
pip install -e . --no-deps          # glmbench importable by the runner, without the core deps
```

flash-attn must be installed before `evo2`: it builds against the installed torch, and `vtx`
imports it at import time without declaring it as a dependency, so `pip install evo2` alone
gives an env that installs cleanly and fails on `import evo2`.

### Pointing the specs at your environments

The shipped specs never hardcode a path. The spec loader expands `${VAR}` and `${VAR:-default}`
in every string, and an unset variable with no default is a loud error naming it:

| Variable | Used by | Default |
| --- | --- | --- |
| `GLMBENCH_PYTHON` | every spec except Evo2 (`runner.python_exe`) | `python`, i.e. the active env |
| `GLMBENCH_EVO2_PYTHON` | `specs/external/evo2-arc-7b-resid.yaml` | `python` |
| `GLMBENCH_LOAM_HF_ROOT` | `specs/loam/LOAM-*.yaml` | the Hub org `Soilytix`; set it to a directory holding `LOAM-25M/` etc. to score local copies |

Run `glmbench` from the activated `glmbench` env and the defaults work for everything but
Evo2. For Evo2, point `GLMBENCH_EVO2_PYTHON` at the `evo2` env's interpreter:

```bash
export GLMBENCH_EVO2_PYTHON="$(conda run -n evo2 python -c 'import sys; print(sys.executable)')"
```

A checkpoint's path is not part of its model hash (the hash covers the weights' content, the
architecture and the adapter version; see [ADD_A_MODEL.md](ADD_A_MODEL.md)), so a model scored
from a local copy lands on the same row as one downloaded from the Hub.

---

## 2. Check the install (CPU only, no weights, no downloads)

```bash
pytest tests/unit
glmbench run --model specs/echo-full.yaml --benchmark dummy_bench --results results/
glmbench models list
```

The unit suite runs on CPU; tests that need torch/transformers or a model download skip when
those are unavailable. The second command runs the model-free `echo` adapter through the
two-task `dummy_bench` benchmark: it exercises the spec loader, the runner subprocess, the wire
protocol, the tasks and the result store, and prints `OK` for both tasks. `glmbench models list`
should show the eight adapters: `echo`, `loam-hf`, `evo2-arc`, `evo`, `glm2`, `genomeocean`,
`prokbert`, `ntv3`.

One environment built by other means showed an import-order crash: `import matplotlib`
*before* `import transformers` segfaulted with no traceback. An environment built as above does
not, but if a script or notebook dies silently on startup, import transformers first (the
notebook already does).

---

## 3. Hugging Face access

Every model downloads from the Hub into your Hugging Face cache (`HF_HOME`, by default
`~/.cache/huggingface`) the first time it runs. The external specs pin a commit; the LOAM
adapter checks the downloaded weights' sha256 against the release's own `loam_export.json`.
Two families are **gated**: accept the terms on each model page, then log in once.

- the LOAM releases, `Soilytix/LOAM-25M`, `-100M`, `-340M`, `-624M`;
- NTv3, `InstaDeepAI/NTv3_100M_pre`.

```bash
hf auth login            # huggingface_hub >= 1.0; older versions: huggingface-cli login
```

The two 7B models are the large downloads (Evo 1.5's weights are about 14 GB in bf16, and
Evo2-7B is of the same order).

---

## 4. Task data

Task data is not redistributed. Three scripts rebuild it from pinned upstream revisions; they
need the `data` extra. Each task checks every data file against the sha256 in its committed
manifest when it loads, and refuses to run on a mismatch.

```bash
python scripts/fetch_bacbench_essentiality.py   # BacBench gene essentiality, ~208 MB
python scripts/fetch_dgeb_ec_dna.py             # DGEB EC classification (DNA), ~1 MB
python scripts/fetch_rnagym_prok.py             # RNAGym prokaryotic DMS (11 assays), ~18 MB
```

The files land under `src/glmbench/tasks/data/`. The RNAGym fetcher also re-emits
`prok_coding_manifest.yaml`; from the pinned sources the rewrite is byte-identical, so a
`git diff` on that file means the data you fetched differs from the paper's.
[TASKS_OVERVIEW.md](TASKS_OVERVIEW.md) describes each task and its source.

---

## 5. Hardware

**GPU.** We recommend 48 GB of GPU memory for the two 7B models (Evo 1.5 and Evo2-7B) with the
specs' batch settings; the LOAM models and the other external models need far less. On a
smaller card, lower `token_budget` / `batch_size` in the spec's `runner.extra`. For LOAM this only moves float noise in the last bits; Evo 1.5's outputs are
not batch-invariant, so for it the change can move the last digits of a score. Choose the GPU
with `CUDA_VISIBLE_DEVICES`, and always give Evo2 exactly one (its package otherwise shards the
model across every visible card).

**Host RAM: the essentiality layer sweep is the heavy job.** `bacbench-essentiality-layer-sweep`
embeds the whole essentiality corpus once at *every* layer and keeps all of it in host memory
while it fits one probe per layer. Plan for about three times *windows × taps × hidden size ×
8 bytes* (the corpus has 164,630 windows):

| Model | Taps × hidden size | Recommended host RAM |
| --- | --- | --- |
| Evo 1.5 (7B) | 33 × 4,096 | over 500 GB |
| Evo2-7B | 17 × 4,096 | about 280 GB |
| LOAM-624M | 17 × 1,792 | about 120 GB |
| LOAM-340M | 13 × 1,536 | about 80 GB |
| LOAM-100M | 9 × 1,024 | about 35 GB |
| LOAM-25M | 6 × 640 | about 15 GB |

So a full sweep of a 7B model needs a large-memory machine, and should not share it with
another memory-hungry job: under memory pressure the sweep keeps running while making no
progress, and it logs nothing per layer. Two ways to size a run first:

- `python scripts/make_essentiality_subset.py --n-train 2 --n-test 1 --out-dir <dir>` builds a
  few-genome subset; point `GLMBENCH_BACBENCH_ESSENTIALITY_MANIFEST` at the manifest it writes,
  run the task, and extrapolate time and memory from the result.
- `GLMBENCH_BACBENCH_LAYER_SWEEP_LAYERS` restricts the sweep (e.g. `"0,8,16"` or `"12-16"`).
  That cuts memory in proportion, but the best layer is then chosen among the listed taps only,
  so the result is not the paper's full sweep.

The EC sweep is small by comparison, but its 128-class probe is CPU-bound: expect minutes per
layer, using every core.

[REPRODUCING_THE_PAPER.md](REPRODUCING_THE_PAPER.md) covers compute per model.
