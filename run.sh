#!/bin/bash

# Train for a 10 sites SU(2) system.
OUTPUT_DIR="result/test"
python main.py \
--output_dir="${OUTPUT_DIR}" \
--system="sun_spin1d" \
--mode="train" \
--n_print=1 \
--n_print_optimization=10 \
--n_save=10 \
--n_iter=100 \
--n_sample=20000 \
--n_sample_initial=1000 \
--n_sample_warmup=10 \
--n_optimize_1=100 \
--n_optimize_2=10 \
--change_n_optimize_at=50 \
--learning_rate_1=1e-3 \
--learning_rate_2=1e-4 \
--change_learning_rate_at=30 \
--n_sites=10 \
--n_spin=2 \
--layers=2 \
--filters=4 \
--kernel=3 \
--network="CNN" \
--state_encoding="one_hot"

# Measure the local energies.
python main.py \
--output_dir="${OUTPUT_DIR}" \
--system="sun_spin1d" \
--mode="energy" \
--n_sample=50000 \
--n_batch=5000 \
--n_sites=10 \
--n_spin=2 \
--layers=2 \
--filters=4 \
--kernel=3 \
--network="CNN" \
--state_encoding="one_hot" \
--load_model_from="${OUTPUT_DIR}/models/latest_model.ckpt-100"

# Measure the loop correlators with various origin_site_index.
for origin_site_index in 0 1 2 3 4 5 6 7 8 9
do
python main.py \
--output_dir="${OUTPUT_DIR}" \
--system="sun_spin1d" \
--mode="loop_correlator" \
--n_sample=50000 \
--n_batch=5000 \
--n_sites=10 \
--n_spin=2 \
--layers=2 \
--filters=4 \
--kernel=3 \
--network="CNN" \
--state_encoding="one_hot" \
--load_model_from="${OUTPUT_DIR}/models/latest_model.ckpt-100" \
--origin_site_index=${origin_site_index}
done
