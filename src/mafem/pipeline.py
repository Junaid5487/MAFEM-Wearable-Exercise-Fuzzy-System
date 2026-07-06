import numpy as np
from .preprocessing import low_pass_filter, min_max_normalize
from .features import extract_features
from .fuzzy import fuzzify, apply_fuzzy_rules, defuzzify_centroid
from .adaptive import StudentProfile

def run_mafem_pipeline_raw(acc_raw, gyr_raw, hr_raw, student_profile, 
                           alpha=0.1, window_size=50, subject_weight_kg=70.0, 
                           subject_age=22, subject_sex='male', sampling_rate=50):
    """
    Runs the full MAFEM pipeline starting from raw multi-axis sensor data.
    
    1. Noise removal via Low-Pass EMA filter.
    2. Min-Max scaling normalization.
    3. Window-based multi-attribute feature extraction.
    4. Fuzzy set allocation and Mamdani rule execution.
    5. Centroid defuzzification and personalization updates.
    
    Returns:
        features_norm (numpy.ndarray): Scaled [0, 1] feature matrix of shape (n_windows, 6)
        results (list of dicts): Window statistics including intensity scores and labels.
    """
    # 1. Apply EMA Filter
    acc_filtered = np.column_stack([low_pass_filter(acc_raw[:, i], alpha) for i in range(3)])
    gyr_filtered = np.column_stack([low_pass_filter(gyr_raw[:, i], alpha) for i in range(3)])
    hr_filtered  = low_pass_filter(hr_raw, alpha)

    # Align streams
    N = min(len(acc_filtered), len(gyr_filtered), len(hr_filtered))
    acc_filtered = acc_filtered[:N]
    gyr_filtered = gyr_filtered[:N]
    hr_filtered  = hr_filtered[:N]
    hr_raw_aligned = hr_raw[:N]

    # 2. Normalize raw sensor signals
    acc_norm = min_max_normalize(acc_filtered)
    gyr_norm = min_max_normalize(gyr_filtered)
    hr_norm  = min_max_normalize(hr_filtered.reshape(-1, 1)).flatten()

    # 3. Feature Extraction (includes Keytel calorie calculations)
    features = extract_features(acc_norm, gyr_norm, hr_norm, hr_raw_aligned,
                                 window_size=window_size,
                                 subject_weight_kg=subject_weight_kg,
                                 subject_age=subject_age,
                                 subject_sex=subject_sex,
                                 sampling_rate=sampling_rate)

    # Store pre-normalized calories for actual output
    calories_raw = features[:, 5].copy()

    # Normalize feature matrix for fuzzification inputs
    features_norm = min_max_normalize(features)

    # 4. Fuzzy Engine & Threshold personalization loop
    results = []
    n_windows = len(features_norm)

    for i in range(n_windows):
        step_c, speed, hrv, cadence, gait, cal_norm = features_norm[i]

        # Calculate mean heart rate in normalized form for the window
        start = i * window_size
        end   = start + window_size
        hr_val = float(np.mean(hr_norm[start:end])) if end <= len(hr_norm) else float(hr_norm[-1])

        cal_kcal = float(calories_raw[i])

        # Fuzzification
        fuzz_vals = {
            'heart_rate':  fuzzify(hr_val,   'heart_rate'),
            'speed':       fuzzify(speed,    'speed'),
            'step_count':  fuzzify(step_c,   'step_count'),
            'hrv':         fuzzify(hrv,      'hrv'),
            'cadence':     fuzzify(cadence,  'cadence'),
            'gait':        fuzzify(gait,     'gait'),
            'calories':    fuzzify(cal_norm, 'calories'),
        }

        # Fuzzy rule evaluation
        rule_out = apply_fuzzy_rules(fuzz_vals)

        # Defuzzification
        crisp, _ = defuzzify_centroid(rule_out)

        # Apply personalized threshold
        th = student_profile.threshold
        if crisp < th - 0.1:
            label = 'LOW'
        elif crisp < th + 0.1:
            label = 'MEDIUM'
        else:
            label = 'HIGH'

        # Update profile stats and thresholds
        student_profile.add_session(crisp, calories_kcal=cal_kcal)

        results.append({
            'window_index': i,
            'crisp': crisp,
            'threshold': th,
            'label': label,
            'calories': cal_kcal,
            'fuzz_values': fuzz_vals,
            'rule_outputs': rule_out
        })

    return features_norm, results
