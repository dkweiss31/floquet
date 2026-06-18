#! /bin/bash

# The purpose is to run the profiling tool with different installed versions.

# Store the base directory path
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_DIR="$BASE_DIR/profile"

# First, run the original version.
cd "$BASE_DIR"
pip uninstall -y floquet
pip install -e .
cd "$PROFILE_DIR"
sudo py-spy record -o origional.json --subprocesses --rate 2 -f speedscope -- python test_floquet.py

# Then, run the tqdm version without new overlap_with_displaced_states.
cd "$BASE_DIR"
pip uninstall -y floquet
cd floquet/utils
mv parallel.py parallel_original.py
cp parallel_modified.py parallel.py
cd "$BASE_DIR"
pip install -e .
cd "$PROFILE_DIR"
sudo py-spy record -o tqdm_with_origional_overlap_with_displaced_states.json --subprocesses --rate 2 -f speedscope -- python test_floquet.py

# Then, run the tqdm version with new overlap_with_displaced_states.
cd "$BASE_DIR"
pip uninstall -y floquet
cd floquet
mv displaced_state.py displaced_state_original.py
cp displaced_state_modified.py displaced_state.py
cd "$BASE_DIR"
pip install -e .
cd "$PROFILE_DIR"
sudo py-spy record -o tqdm_with_new_overlap_with_displaced_states.json --subprocesses --rate 2 -f speedscope -- python test_floquet.py

# Cleanup - restore original files
cd "$BASE_DIR/floquet/utils"
mv parallel_original.py parallel.py
cd "$BASE_DIR/floquet"
mv displaced_state_original.py displaced_state.py