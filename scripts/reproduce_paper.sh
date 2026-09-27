#!/usr/bin/env bash
# Reproduce the LOAM paper's gLMBench rows for one model, or for all 13: fetch the task data
# (once), run the `loam_paper` benchmark, and compare the result with paper/.
#
#   scripts/reproduce_paper.sh --model LOAM-25M            # the smallest and quickest
#   scripts/reproduce_paper.sh --model "GenomeOcean-100M"
#   scripts/reproduce_paper.sh --spec specs/loam/LOAM-624M.yaml
#   scripts/reproduce_paper.sh --all                       # all 13 models, one after another
#   scripts/reproduce_paper.sh --list                      # the model ids and their specs
#
# Options:
#   --gpu N          physical GPU (exported as CUDA_VISIBLE_DEVICES; default 0)
#   --results DIR    where records go (default: results)
#   --exact          require bitwise agreement with the paper (the reference environment gives it;
#                    elsewhere the per-task tolerances of compare_to_paper.py apply)
#   --skip-fetch     do not fetch task data
#   --no-resume      recompute tasks that an earlier record of the same model hash has
#   --dry-run        print the runner commands only
#
# Environment:
#   GLMBENCH_PYTHON       python with glmbench + torch + transformers (default: python)
#   GLMBENCH_EVO2_PYTHON  python of the Evo 2 environment (needed for Evo2-7B only)
#
# Compute: the essentiality embedding pass dominates and grows with model size; the 7B models
# take by far the longest. The essentiality layer sweep keeps every layer's pooled embeddings in
# host RAM; the 7B models need several hundred GB. See docs/REPRODUCING_THE_PAPER.md.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."   # repository root

GPU="${GPU:-0}"
RESULTS="results"
MODEL=""; SPEC=""; ALL=""; LIST=""
EXACT=""; SKIP_FETCH=""; NO_RESUME=""; DRY_RUN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) shift; MODEL="${1:?--model needs a model id}"; shift ;;
    --spec) shift; SPEC="${1:?--spec needs a path}"; shift ;;
    --all) ALL=1; shift ;;
    --list) LIST=1; shift ;;
    --gpu) shift; GPU="${1:?--gpu needs an index}"; shift ;;
    --results) shift; RESULTS="${1:?--results needs a path}"; shift ;;
    --exact) EXACT="--exact"; shift ;;
    --skip-fetch) SKIP_FETCH=1; shift ;;
    --no-resume) NO_RESUME="--no-resume"; shift ;;
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
export CUDA_VISIBLE_DEVICES="$GPU"
# A dry run still writes a record (every task ERROR, since nothing runs); keep it out of the
# real results store.
if [[ -n "$DRY_RUN" ]]; then
  RESULTS="$(mktemp -d)/results"
fi
PYTHON="${GLMBENCH_PYTHON:-python}"
"$PYTHON" -c "import glmbench" 2>/dev/null || {
  echo "ERROR: '$PYTHON' cannot import glmbench. Activate the environment where you ran" >&2
  echo "       'pip install -e .', or set GLMBENCH_PYTHON=/path/to/python." >&2
  exit 1
}

log() { printf '\n\033[1;34m[reproduce]\033[0m %s\n' "$*"; }

# model id -> spec, from paper/models.csv
spec_of() {
  "$PYTHON" - "$1" <<'PY'
import csv, sys
rows = {r["model_id"]: r["spec"] for r in csv.DictReader(open("paper/models.csv"))}
if sys.argv[1] not in rows:
    sys.exit(f"unknown model id {sys.argv[1]!r}; known: {', '.join(rows)}")
print(rows[sys.argv[1]])
PY
}

if [[ -n "$LIST" ]]; then
  "$PYTHON" -c "import csv; [print(f\"{r['model_id']:<28} {r['spec']}\") for r in csv.DictReader(open('paper/models.csv'))]"
  exit 0
fi

SPECS=()
if [[ -n "$ALL" ]]; then
  mapfile -t SPECS < <("$PYTHON" -c "import csv; [print(r['spec']) for r in csv.DictReader(open('paper/models.csv'))]")
elif [[ -n "$MODEL" ]]; then
  SPECS=("$(spec_of "$MODEL")")
elif [[ -n "$SPEC" ]]; then
  SPECS=("$SPEC")
else
  echo "give --model ID, --spec PATH or --all (see --help)" >&2; exit 2
fi

# --- task data (once) --------------------------------------------------------------------
if [[ -z "$SKIP_FETCH" && -z "$DRY_RUN" ]]; then
  DATA="src/glmbench/tasks/data"
  FETCHES=(
    "$DATA/bacbench_essentiality/essential_genes_test.csv|scripts/fetch_bacbench_essentiality.py"
    "$DATA/dgeb_ec_dna/ec_classification_dna_test.csv|scripts/fetch_dgeb_ec_dna.py"
    "$DATA/rnagym/processed_DMS_files/BLAT_ECOLX_Firnberg_2014.csv|scripts/fetch_rnagym_prok.py"
  )
  for entry in "${FETCHES[@]}"; do
    sentinel="${entry%%|*}"; script="${entry##*|}"
    if [[ ! -f "$sentinel" ]]; then
      log "fetching task data: $script"
      "$PYTHON" "$script"
    fi
  done
fi

# --- run + compare ---------------------------------------------------------------------------
STATUS=0
for spec in "${SPECS[@]}"; do
  log "$spec -> $RESULTS/loam_paper/ (GPU $GPU)"
  "$PYTHON" -m glmbench.cli run --model "$spec" --benchmark loam_paper --results "$RESULTS" \
    $NO_RESUME $DRY_RUN
  [[ -n "$DRY_RUN" ]] && continue
  hash="$("$PYTHON" - "$spec" <<'PY'
import sys
from glmbench import registry
import glmbench.adapters  # noqa: F401
from glmbench.config.model_spec import load_model_spec
spec = load_model_spec(sys.argv[1])
print(registry.resolve("adapter", spec.adapter).from_spec(spec, dry_run=True).model_hash())
PY
)"
  record="$RESULTS/loam_paper/glmb_${hash#glmb:}.json"
  log "comparing $record with the paper"
  "$PYTHON" scripts/compare_to_paper.py "$record" $EXACT || STATUS=1
done

[[ -z "$DRY_RUN" ]] && "$PYTHON" -m glmbench.cli leaderboard --benchmark loam_paper --results "$RESULTS"
exit $STATUS
