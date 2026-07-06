import sys
import os
import numpy as np
import pytest

# Add src to PATH so tests can import local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mafem.preprocessing import low_pass_filter, min_max_normalize
from mafem.features import estimate_calories, extract_features
from mafem.fuzzy import triangular_mf, fuzzify, apply_fuzzy_rules, defuzzify_centroid
from mafem.adaptive import StudentProfile
from mafem.pipeline import run_mafem_pipeline_raw

def test_low_pass_filter():
    signal = np.array([1.0, 1.0, 1.0, 1.0])
    filtered = low_pass_filter(signal, alpha=0.1)
    
    assert len(filtered) == 4
    # With a constant signal, EMA should remain constant (or close to it)
    assert np.allclose(filtered, 1.0)
    
    # Test empty signal
    assert len(low_pass_filter(np.array([]))) == 0

def test_min_max_normalize():
    data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    norm = min_max_normalize(data)
    
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert np.all((norm >= 0.0) & (norm <= 1.0))
    
    # Test flat data (avoid divide by zero)
    flat_data = np.array([5.0, 5.0, 5.0])
    norm_flat = min_max_normalize(flat_data)
    assert np.all(norm_flat == 0.0)

def test_estimate_calories():
    # Male:   (-55.0969 + 0.6309*HR + 0.1988*W + 0.2017*A) / 4.184
    # Female: (-20.4022 + 0.4472*HR - 0.1263*W + 0.0740*A) / 4.184
    
    duration = 60.0 # 60 seconds = 1 minute
    cal_male = estimate_calories(hr_mean_bpm=120.0, duration_sec=duration, weight_kg=70.0, age=22, sex='male')
    cal_female = estimate_calories(hr_mean_bpm=120.0, duration_sec=duration, weight_kg=70.0, age=22, sex='female')
    
    # Check that outputs are floating point and positive
    assert cal_male > 0.0
    assert cal_female > 0.0
    
    # At 120 bpm, male calories should be different from female due to equations
    assert cal_male != cal_female
    
    # Higher HR should yield more calories
    cal_male_higher = estimate_calories(hr_mean_bpm=160.0, duration_sec=duration, weight_kg=70.0, age=22, sex='male')
    assert cal_male_higher > cal_male

def test_triangular_mf():
    # Test boundary limits
    assert triangular_mf(-0.5, 0.0, 0.5, 1.0) == 0.0
    assert triangular_mf(1.5, 0.0, 0.5, 1.0) == 0.0
    
    # Test peak value
    assert triangular_mf(0.5, 0.0, 0.5, 1.0) == 1.0
    
    # Test linear slopes
    assert triangular_mf(0.25, 0.0, 0.5, 1.0) == 0.5
    assert triangular_mf(0.75, 0.0, 0.5, 1.0) == 0.5

def test_fuzzy_rules_and_defuzzification():
    # Setup mock fuzzified values
    fuzz_vals = {
        'heart_rate': {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 1.0},
        'speed':      {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 1.0},
        'step_count': {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 1.0},
        'hrv':        {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 1.0},
        'cadence':    {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 1.0},
        'gait':       {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 1.0},
        'calories':   {'LOW': 0.0, 'MEDIUM': 0.0, 'HIGH': 1.0}
    }
    
    rules = apply_fuzzy_rules(fuzz_vals)
    assert rules['HIGH'] == 1.0
    assert rules['LOW'] == 0.0
    
    crisp, label = defuzzify_centroid(rules)
    assert crisp == 0.85 # HIGH output center
    assert label == 'HIGH'

def test_adaptive_threshold_personalization():
    student = StudentProfile(student_id="test_student", initial_fitness=0.5, K=0.1)
    
    # Log 5 windows of high intensity (0.85 score)
    for _ in range(5):
        student.add_session(0.85, calories_kcal=0.05)
        
    # The threshold should adjust upwards since current fitness (avg of last 5 scores = 0.85)
    # is greater than the initial fitness baseline (0.50).
    assert student.threshold > 0.50
    assert student.threshold <= 0.90
    assert student.total_calories_burned == pytest.approx(0.25)
