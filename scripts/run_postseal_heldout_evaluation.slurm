#!/usr/bin/bash -p
#SBATCH --job-name=c4-postseal-eval
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --exclude=gpu-05,gpu-50,gpu-51,gpu-56
#SBATCH --output=c4-postseal-eval-%j.log

set -euo pipefail
umask 077
readonly PATH=/usr/bin:/bin:/sbin
export PATH
unset BASH_ENV ENV LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS

: "${CONNECT4_RUNTIME_ROOT:?required}"
: "${CONNECT4_RUNTIME_MANIFEST_SHA256:?required}"
: "${CONNECT4_SLURM_LAUNCHER_SHA256:?required}"

hash_file() {
  local source="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${source}" | awk '{print $1}'
  else
    shasum -a 256 "${source}" | awk '{print $1}'
  fi
}

file_mode() {
  local source="$1"
  if stat -c '%a' "${source}" >/dev/null 2>&1; then
    stat -c '%a' "${source}"
  else
    stat -f '%Lp' "${source}"
  fi
}

file_links() {
  local source="$1"
  if stat -c '%h' "${source}" >/dev/null 2>&1; then
    stat -c '%h' "${source}"
  else
    stat -f '%l' "${source}"
  fi
}

require_sha256() {
  local value="$1"
  local label="$2"
  if [[ ! "${value}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "${label} must be a lowercase full SHA-256" >&2
    exit 2
  fi
}

require_sha256 "${CONNECT4_RUNTIME_MANIFEST_SHA256}" "runtime manifest pin"
require_sha256 "${CONNECT4_SLURM_LAUNCHER_SHA256}" "Slurm launcher pin"
readonly running_launcher="${BASH_SOURCE[0]}"
if [[ ! -f "${running_launcher}" || -L "${running_launcher}" ]]; then
  echo "Slurm launcher must be a regular non-symlink file" >&2
  exit 2
fi
if [[ "$(hash_file "${running_launcher}")" != "${CONNECT4_SLURM_LAUNCHER_SHA256}" ]]; then
  echo "Slurm launcher SHA-256 differs from its external pin" >&2
  exit 2
fi

if [[ "${CONNECT4_RUNTIME_ROOT}" != /* || -L "${CONNECT4_RUNTIME_ROOT}" ]]; then
  echo "runtime root must be an absolute non-symlink lexical path" >&2
  exit 2
fi
readonly runtime_root="$(cd "${CONNECT4_RUNTIME_ROOT}" && pwd -P)"
if [[ "${runtime_root}" != "${CONNECT4_RUNTIME_ROOT}" || "$(file_mode "${runtime_root}")" != "555" ]]; then
  echo "runtime root aliases another path or is not immutable mode 0555" >&2
  exit 2
fi
readonly runtime_manifest="${runtime_root}/postseal_runtime.sha256"
if [[ ! -f "${runtime_manifest}" || -L "${runtime_manifest}" || "$(file_links "${runtime_manifest}")" != "1" || "$(file_mode "${runtime_manifest}")" != "444" ]]; then
  echo "runtime manifest must be one regular non-symlink file" >&2
  exit 2
fi
if [[ "$(hash_file "${runtime_manifest}")" != "${CONNECT4_RUNTIME_MANIFEST_SHA256}" ]]; then
  echo "runtime manifest SHA-256 differs from its external pin" >&2
  exit 2
fi

readonly expected_runtime_files="architecture_contract.py
eval/__init__.py
eval/postseal.py
eval/postseal_execution.py
eval/postseal_metrics.py
eval/quality.py
scripts/__init__.py
scripts/check_4d_quality.py
scripts/evaluate_postseal_heldout.py
scripts/run_postseal_heldout_evaluation.slurm
scripts/visualize_4d_comparison.py
scripts/visualize_texture_audit.py
utils/__init__.py
utils/source_provenance.py
utils/spatial_detail.py"
manifest_paths=""
runtime_launcher_sha256=""
while IFS= read -r binding || [[ -n "${binding}" ]]; do
  if [[ ! "${binding}" =~ ^([0-9a-f]{64})\ \ ([A-Za-z0-9_./-]+)$ ]]; then
    echo "runtime manifest has a malformed binding" >&2
    exit 2
  fi
  digest="${BASH_REMATCH[1]}"
  relative="${BASH_REMATCH[2]}"
  if [[ "${relative}" == /* || "${relative}" == *".."* ]]; then
    echo "runtime manifest path is unsafe" >&2
    exit 2
  fi
  source="${runtime_root}/${relative}"
  if [[ ! -f "${source}" || -L "${source}" || "$(file_links "${source}")" != "1" || "$(file_mode "${source}")" != "444" ]]; then
    echo "runtime file is missing, linked, or mutable: ${relative}" >&2
    exit 2
  fi
  if [[ "$(hash_file "${source}")" != "${digest}" ]]; then
    echo "runtime file SHA-256 differs: ${relative}" >&2
    exit 2
  fi
  if [[ "${relative}" == "scripts/run_postseal_heldout_evaluation.slurm" ]]; then
    runtime_launcher_sha256="${digest}"
  fi
  manifest_paths+="${relative}"$'\n'
done < "${runtime_manifest}"
manifest_paths="${manifest_paths%$'\n'}"
if [[ "${manifest_paths}" != "${expected_runtime_files}" ]]; then
  echo "runtime manifest file set/order differs from the closed contract" >&2
  exit 2
fi
if [[ "${runtime_launcher_sha256}" != "${CONNECT4_SLURM_LAUNCHER_SHA256}" ]]; then
  echo "running Slurm launcher differs from the staged runtime launcher" >&2
  exit 2
fi
if [[ -n "$(find "${runtime_root}" -type l -print -quit)" ]]; then
  echo "runtime tree contains a symlink" >&2
  exit 2
fi
while IFS= read -r directory; do
  if [[ "$(file_mode "${directory}")" != "555" ]]; then
    echo "runtime directory is mutable: ${directory}" >&2
    exit 2
  fi
done < <(find "${runtime_root}" -type d -print)
readonly observed_runtime_files="$(
  find "${runtime_root}" -type f -print \
    | sed "s#^${runtime_root}/##" \
    | LC_ALL=C sort
)"
readonly expected_runtime_tree="$(
  { printf '%s\n' "${expected_runtime_files}"; printf '%s\n' 'postseal_runtime.sha256'; } \
    | LC_ALL=C sort
)"
if [[ "${observed_runtime_files}" != "${expected_runtime_tree}" ]]; then
  echo "runtime tree has a missing or extra file" >&2
  exit 2
fi
export CONNECT4_RUNTIME_PREIMPORT_AUTHENTICATED=1

if [[ "${CONNECT4_RUNTIME_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf '%s\n' "authenticated runtime ${CONNECT4_RUNTIME_MANIFEST_SHA256}"
  exit 0
fi

: "${CONNECT4_PREDICTION_ROOT:?required}"
: "${CONNECT4_PYTHON_SHA256:?required}"
: "${CONNECT4_DEPENDENCY_RUNTIME_AUTHORITY:?required}"
: "${CONNECT4_DEPENDENCY_RUNTIME_AUTHORITY_SHA256:?required}"
: "${CONNECT4_EXECUTION_AUTHORITY:?required}"
: "${CONNECT4_EXECUTION_AUTHORITY_SHA256:?required}"
: "${CONNECT4_PREDICTION_SET_SEAL_SHA256:?required}"
: "${CONNECT4_SYNTHESIS_CHECKPOINT_SHA256:?required}"
: "${CONNECT4_TRAINING_RUN_ARTIFACT_IDENTITY_SHA256:?required}"
: "${CONNECT4_TRAINING_TARGET_ARTIFACT_IDENTITIES_SHA256:?required}"
: "${CONNECT4_SPATIAL_AUTHORITY_SHA256:?required}"
: "${CONNECT4_STAGE_B_COMPLETED_SET:?required}"
: "${CONNECT4_STAGE_B_COMPLETED_SET_SHA256:?required}"
: "${CONNECT4_POSTSEAL_EVALUATION_ROOT:?required}"

: "${CONNECT4_PYTHON:?required}"
readonly python_entry="${CONNECT4_PYTHON}"
readonly python="$(/usr/bin/readlink -f "${python_entry}")"
readonly evaluator="${runtime_root}/scripts/evaluate_postseal_heldout.py"
if [[ -z "${python}" || "${python}" != /* || -L "${python}" || ! -x "${python}" || ! -f "${evaluator}" || -L "${evaluator}" ]]; then
  echo "pinned Python or post-seal evaluator is unavailable" >&2
  exit 2
fi
require_sha256 "${CONNECT4_PYTHON_SHA256}" "Python executable pin"
if [[ "$(hash_file "${python}")" != "${CONNECT4_PYTHON_SHA256}" ]]; then
  echo "Python executable SHA-256 differs from its external pin" >&2
  exit 2
fi
if [[ "$(hostname)" == *login* || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "post-seal evaluation must run inside a GPU compute allocation" >&2
  exit 2
fi

exec "${python}" -B -I -S "${evaluator}" evaluate \
  --prediction-root "${CONNECT4_PREDICTION_ROOT}" \
  --prediction-set-seal-sha256 "${CONNECT4_PREDICTION_SET_SEAL_SHA256}" \
  --synthesis-checkpoint-sha256 "${CONNECT4_SYNTHESIS_CHECKPOINT_SHA256}" \
  --training-run-artifact-identity-sha256 "${CONNECT4_TRAINING_RUN_ARTIFACT_IDENTITY_SHA256}" \
  --training-target-artifact-identities-sha256 "${CONNECT4_TRAINING_TARGET_ARTIFACT_IDENTITIES_SHA256}" \
  --spatial-authority-sha256 "${CONNECT4_SPATIAL_AUTHORITY_SHA256}" \
  --stage-b-completed-set "${CONNECT4_STAGE_B_COMPLETED_SET}" \
  --stage-b-completed-set-sha256 "${CONNECT4_STAGE_B_COMPLETED_SET_SHA256}" \
  --runtime-manifest "${runtime_manifest}" \
  --runtime-manifest-sha256 "${CONNECT4_RUNTIME_MANIFEST_SHA256}" \
  --slurm-launcher-sha256 "${CONNECT4_SLURM_LAUNCHER_SHA256}" \
  --python-executable-sha256 "${CONNECT4_PYTHON_SHA256}" \
  --dependency-runtime-authority "${CONNECT4_DEPENDENCY_RUNTIME_AUTHORITY}" \
  --dependency-runtime-authority-sha256 "${CONNECT4_DEPENDENCY_RUNTIME_AUTHORITY_SHA256}" \
  --execution-authority "${CONNECT4_EXECUTION_AUTHORITY}" \
  --execution-authority-sha256 "${CONNECT4_EXECUTION_AUTHORITY_SHA256}" \
  --output-root "${CONNECT4_POSTSEAL_EVALUATION_ROOT}"
