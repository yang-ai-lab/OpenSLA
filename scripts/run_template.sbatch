#!/usr/bin/env bash
#SBATCH --job-name=OpenSLA
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=OpenSLA-%j.out

set -euo pipefail

# Slurm copies batch scripts; submit from the repository or set OPENSLA_REPO.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    root="${OPENSLA_REPO:-${SLURM_SUBMIT_DIR:?}}"
else
    root="${OPENSLA_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
cd -- "$root"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -m opensla.cli "$@"
