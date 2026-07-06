import numpy as np
from scipy.signal import find_peaks

def estimate_calories(hr_mean_bpm, duration_sec, weight_kg=70.0, age=22, sex='male'):
    """
    Calorie estimation per window using the Keytel et al. (2005) heart-rate-based formula.
    
    Male:   kcal/min = (-55.0969 + 0.6309*HR + 0.1988*W + 0.2017*A) / 4.184
    Female: kcal/min = (-20.4022 + 0.4472*HR - 0.1263*W + 0.0740*A) / 4.184
    """
    duration_min = duration_sec / 60.0
    HR, W, A = hr_mean_bpm, weight_kg, age

    if str(sex).lower() == 'female':
        kcal_per_min = (-20.4022 + 0.4472 * HR - 0.1263 * W + 0.0740 * A) / 4.184
    else:
        kcal_per_min = (-55.0969 + 0.6309 * HR + 0.1988 * W + 0.2017 * A) / 4.184

    calories = max(0.0, kcal_per_min * duration_min)  # clamp to non-negative
    return calories

def extract_features(acc, gyr, hr, hr_raw, window_size=50,
                     subject_weight_kg=70.0, subject_age=22, subject_sex='male',
                     sampling_rate=50):
    """
    Extract 6 features per window from raw/normalized sensor data.
    Features:
      1. Step Count        - peak detection on acc magnitude
      2. Speed             - integration of acc magnitude over window
      3. HRV               - std deviation of HR in window
      4. Cadence           - steps per minute
      5. Gait              - angular velocity magnitude from gyroscope
      6. Calories Burned   - Keytel HR-based kcal estimate for the window
    """
    if acc is None or gyr is None or hr is None or hr_raw is None:
        raise ValueError("All sensor inputs (acc, gyr, hr, hr_raw) must be provided.")
        
    features = []
    n_windows = len(acc) // window_size
    duration_sec = window_size / sampling_rate

    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size

        acc_window    = acc[start:end]
        gyr_window    = gyr[start:end]
        hr_raw_window = hr_raw[start:end]

        # Magnitude of acceleration
        acc_mag = np.linalg.norm(acc_window, axis=1)

        # 1. Step Count - peaks in acc magnitude
        peaks, _ = find_peaks(acc_mag, height=0.5, distance=5)
        step_count = len(peaks)

        # 2. Speed - cumulative sum of acc magnitude (proxy)
        if hasattr(np, 'trapezoid'):
            speed = np.trapezoid(acc_mag) / window_size
        else:
            speed = np.trapz(acc_mag) / window_size

        # 3. HRV - std of heart rate in window
        hrv = np.std(hr_raw_window)

        # 4. Cadence - steps per minute
        cadence = (step_count / duration_sec) * 60

        # 5. Gait - mean angular velocity magnitude
        gyr_mag = np.linalg.norm(gyr_window, axis=1)
        gait = np.mean(gyr_mag)

        # 6. Calories - Keytel formula using mean raw HR for this window
        hr_mean_bpm = float(np.mean(hr_raw_window))
        calories = estimate_calories(hr_mean_bpm, duration_sec,
                                     weight_kg=subject_weight_kg,
                                     age=subject_age, sex=subject_sex)

        features.append([step_count, speed, hrv, cadence, gait, calories])

    return np.array(features)
