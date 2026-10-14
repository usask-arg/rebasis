#!/usr/bin/env bash
# Sampling-study batch: several months and levels, three covariance models.
set -u
cd "$(dirname "$0")/../.."
for c in "2010 9 46" "2010 9 10" "2010 3 46" "2010 11 46"; do
  for mode in "" "--local 20" "--two-scale"; do
    echo "===== $c $mode"
    # shellcheck disable=SC2086
    uv run --with h5py --with h5netcdf python examples/osiris/sampling_study.py $c --knots 15 $mode 2>&1 \
      | grep -E "RMS|truth within|fitted|Error|error"
  done
done
