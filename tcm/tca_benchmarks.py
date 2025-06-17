# benchmarks.py
import numpy as np
from tcm.tcm_model import TCM_A

def run_recall_benchmark(params, item_pool_size, list_length, distractor_dur=0, time_limit=1000):
    """
    Runs a single free recall trial and prints the results.
    """
    print(f"\n--- Running Recall Benchmark ---")
    print(f"Distractor Duration: {distractor_dur} steps")

    # Instantiate the model
    model = TCM_A(item_pool_size, item_pool_size, params)
    
    # Select a random list of items to study
    study_list_indices = np.random.choice(item_pool_size, list_length, replace=False)
    print(f"Study List (indices): {study_list_indices}")

    # Encode the list
    model.encode_list(study_list_indices, distractor_duration=distractor_dur)
    
    # Perform recall
    recalled_indices = model.recall(list_length, time_limit=time_limit)
    print(f"Recalled List (indices): {recalled_indices}")
    
    correct_recalls = len(set(recalled_indices) & set(study_list_indices))
    print(f"Performance: {correct_recalls} / {list_length} items recalled correctly.")
    print("-" * 30)

def run_recognition_benchmark(params, item_pool_size, list_length, num_lures=10):
    """
    Runs a recognition test and prints the results.
    """
    print(f"\n--- Running Recognition Benchmark ---")
    
    model = TCM_A(item_pool_size, item_pool_size, params)
    
    # Define targets and lures
    all_indices = np.arange(item_pool_size)
    targets = np.random.choice(all_indices, list_length, replace=False)
    remaining_indices = np.setdiff1d(all_indices, targets)
    lures = np.random.choice(remaining_indices, num_lures, replace=False)
    
    print(f"Studied Targets (indices): {targets}")
    
    # Encode the target list
    model.encode_list(targets)
    
    # Test recognition for targets (should be "old")
    hits = 0
    for item_idx in targets:
        _, recognized = model.recognize(item_idx)
        if recognized:
            hits += 1
            
    # Test recognition for lures (should be "new")
    false_alarms = 0
    for item_idx in lures:
        _, recognized = model.recognize(item_idx)
        if recognized:
            false_alarms += 1

    print(f"Recognition Performance:")
    print(f"  Hits: {hits} / {len(targets)} ({(hits/len(targets)):.2%})")
    print(f"  False Alarms: {false_alarms} / {len(lures)} ({(false_alarms/len(lures)):.2%})")
    print("-" * 30)


if __name__ == "__main__":
    # Define parameters for the model based on principles from the research [1, 1, 1]
    tcm_params = {
        'beta_enc': 0.5,         # Context drift rate during encoding
        'beta_rec': 0.5,         # Context drift rate during recall
        'beta_dist': 0.9,        # Context drift during distractor
        'gamma_ft': 0.5,         # Weighting of experimental vs. pre-experimental item-to-context
        'gamma_tf': 0.8,         # Weighting of experimental vs. pre-experimental context-to-item
        'phi_s': 1.5,            # Primacy effect scale
        'phi_d': 1.0,            # Primacy effect decay
        'kappa': 0.1,            # Accumulator leak rate
        'lambda': 0.05,          # Lateral inhibition
        'sigma': 0.02,           # Accumulator noise
        'tau': 0.1,              # Accumulator time constant
        'threshold': 1.0,        # Recall threshold
        'recog_threshold': 0.1   # Recognition threshold
    }

    ITEM_POOL = 50
    LIST_LENGTH = 12

    # --- Recall Benchmarks ---
    # Immediate Free Recall (no distractor)
    run_recall_benchmark(tcm_params, ITEM_POOL, LIST_LENGTH, distractor_dur=0)
    
    # Delayed Free Recall (16 steps of distractor)
    run_recall_benchmark(tcm_params, ITEM_POOL, LIST_LENGTH, distractor_dur=16)

    # --- Recognition Benchmark ---
    run_recognition_benchmark(tcm_params, ITEM_POOL, LIST_LENGTH, num_lures=LIST_LENGTH)
