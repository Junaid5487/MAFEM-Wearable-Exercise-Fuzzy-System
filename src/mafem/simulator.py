import numpy as np

def generate_simulated_workout(workout_type='Cardio HIIT', sampling_rate=50):
    """
    Generates a mock workout session (accelerometer, gyroscope, heart rate, labels)
    with realistic movement patterns, noise, and physiological changes.
    
    Supported workout_types:
      * 'Cardio HIIT': high-intensity multi-activity session.
      * 'Strength Training': focused on squats and push-ups with slower heart rate changes.
      * 'Active Recovery': slow movements (walking) and recovery resting.
    """
    np.random.seed(42)
    
    if workout_type == 'Strength Training':
        stages = [
            {"name": "rest",     "duration": 15, "hr_start": 70.0,  "hr_end": 72.0},
            {"name": "squats",   "duration": 40, "hr_start": 72.0,  "hr_end": 115.0},
            {"name": "rest",     "duration": 20, "hr_start": 115.0, "hr_end": 95.0},
            {"name": "push_ups", "duration": 40, "hr_start": 95.0,  "hr_end": 135.0},
            {"name": "rest",     "duration": 20, "hr_start": 135.0, "hr_end": 105.0}
        ]
    elif workout_type == 'Active Recovery':
        stages = [
            {"name": "rest",    "duration": 20, "hr_start": 68.0, "hr_end": 70.0},
            {"name": "walking", "duration": 60, "hr_start": 70.0, "hr_end": 95.0},
            {"name": "rest",    "duration": 30, "hr_start": 95.0, "hr_end": 78.0}
        ]
    else:  # Cardio HIIT
        stages = [
            {"name": "rest",          "duration": 15, "hr_start": 72.0,  "hr_end": 75.0},
            {"name": "squats",        "duration": 30, "hr_start": 75.0,  "hr_end": 120.0},
            {"name": "rest",          "duration": 15, "hr_start": 120.0, "hr_end": 100.0},
            {"name": "jumping_jacks", "duration": 30, "hr_start": 100.0, "hr_end": 165.0},
            {"name": "push_ups",      "duration": 20, "hr_start": 165.0, "hr_end": 145.0},
            {"name": "rest",          "duration": 15, "hr_start": 145.0, "hr_end": 110.0}
        ]
        
    acc_list = []
    gyr_list = []
    hr_list = []
    labels_list = []
    
    for stage in stages:
        n_samples = stage["duration"] * sampling_rate
        t = np.arange(n_samples) / sampling_rate
        
        # 1. Base Heart Rate interpolation
        hr_block = np.linspace(stage["hr_start"], stage["hr_end"], n_samples)
        # Add slight heart rate variability (HRV) noise
        hr_block += np.random.normal(0, 1.5, n_samples)
        
        # Initialize sensor data for this block
        acc_block = np.zeros((n_samples, 3))
        gyr_block = np.zeros((n_samples, 3))
        
        name = stage["name"]
        
        if name == "rest":
            # Gravity on Z axis primarily (wrist horizontal/resting)
            acc_block[:, 0] = np.random.normal(0, 0.05, n_samples)
            acc_block[:, 1] = np.random.normal(0, 0.05, n_samples)
            acc_block[:, 2] = 1.0 + np.random.normal(0, 0.05, n_samples)
            
            # Minor gyroscope drift/jitter
            gyr_block = np.random.normal(0, 5.0, (n_samples, 3))
            
        elif name == "squats":
            # Squats have periodic vertical and torso lean components
            freq = 0.3
            acc_block[:, 0] = np.random.normal(0, 0.08, n_samples)
            acc_block[:, 1] = 0.2 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.08, n_samples)
            acc_block[:, 2] = 1.0 + 0.3 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.08, n_samples)
            
            gyr_block[:, 0] = 30.0 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 10.0, n_samples)
            gyr_block[:, 1] = np.random.normal(0, 10.0, n_samples)
            gyr_block[:, 2] = 15.0 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 10.0, n_samples)
            
        elif name == "jumping_jacks":
            # Jumping jacks are high impact, rapid movement
            freq = 0.9
            acc_block[:, 0] = 1.2 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.25, n_samples)
            acc_block[:, 1] = 0.8 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.25, n_samples)
            acc_block[:, 2] = 1.0 + 1.5 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.25, n_samples)
            
            gyr_block[:, 0] = 120.0 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 20.0, n_samples)
            gyr_block[:, 1] = 80.0 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 20.0, n_samples)
            gyr_block[:, 2] = 100.0 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 20.0, n_samples)
            
        elif name == "push_ups":
            freq = 0.4
            acc_block[:, 0] = 0.5 + np.random.normal(0, 0.04, n_samples)
            acc_block[:, 1] = 0.8 + 0.15 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.04, n_samples)
            acc_block[:, 2] = 0.1 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.04, n_samples)
            gyr_block = np.random.normal(0, 3.0, (n_samples, 3))
            
        elif name == "walking":
            # Walking is a steady, lower impact periodic movement
            freq = 0.8
            acc_block[:, 0] = 0.15 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05, n_samples)
            acc_block[:, 1] = np.random.normal(0, 0.05, n_samples)
            acc_block[:, 2] = 1.0 + 0.10 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.05, n_samples)
            
            gyr_block[:, 0] = 20.0 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 5.0, n_samples)
            gyr_block[:, 1] = np.random.normal(0, 5.0, n_samples)
            gyr_block[:, 2] = 10.0 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 5.0, n_samples)
            
        acc_list.append(acc_block)
        gyr_list.append(gyr_block)
        hr_list.append(hr_block)
        labels_list.append([name] * n_samples)
        
    acc = np.vstack(acc_list)
    gyr = np.vstack(gyr_list)
    hr = np.concatenate(hr_list)
    labels = np.concatenate(labels_list)
    
    return acc, gyr, hr, labels
