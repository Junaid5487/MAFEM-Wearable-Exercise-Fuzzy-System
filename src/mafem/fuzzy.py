import numpy as np
import os
import json

# Locate the config file relative to this script's directory
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    'config', 
    'fuzzy_config.json'
)

# Load configuration parameters with fallback defaults
try:
    with open(CONFIG_PATH, 'r') as f:
        _config = json.load(f)
    MF_PARAMS = _config["MF_PARAMS"]
    OUTPUT_CENTRES = _config["OUTPUT_CENTRES"]
except Exception:
    # Fallback to default paper parameters if config load fails
    MF_PARAMS = {
        'heart_rate': {
            'LOW':    [0.0, 0.3, 0.5],
            'MEDIUM': [0.4, 0.6, 0.8],
            'HIGH':   [0.7, 0.9, 1.0]
        },
        'speed': {
            'LOW':    [0.0, 0.2, 0.4],
            'MEDIUM': [0.3, 0.5, 0.7],
            'HIGH':   [0.6, 0.8, 1.0]
        },
        'step_count': {
            'LOW':    [0.0, 0.2, 0.4],
            'MEDIUM': [0.3, 0.5, 0.7],
            'HIGH':   [0.6, 0.8, 1.0]
        },
        'hrv': {
            'LOW':    [0.0, 0.2, 0.4],
            'MEDIUM': [0.3, 0.5, 0.7],
            'HIGH':   [0.6, 0.8, 1.0]
        },
        'cadence': {
            'LOW':    [0.0, 0.2, 0.4],
            'MEDIUM': [0.3, 0.5, 0.7],
            'HIGH':   [0.6, 0.8, 1.0]
        },
        'gait': {
            'LOW':    [0.0, 0.2, 0.4],
            'MEDIUM': [0.3, 0.5, 0.7],
            'HIGH':   [0.6, 0.8, 1.0]
        },
        'calories': {
            'LOW':    [0.0, 0.2, 0.4],
            'MEDIUM': [0.3, 0.5, 0.7],
            'HIGH':   [0.6, 0.8, 1.0]
        }
    }
    OUTPUT_CENTRES = {'LOW': 0.15, 'MEDIUM': 0.5, 'HIGH': 0.85}

def triangular_mf(x, a, b, c):
    """
    Triangular membership function.
    Formula:
      0           if x <= a
      (x-a)/(b-a) if a < x <= b
      (c-x)/(c-b) if b < x <= c
      0           if x >= c
    """
    x_clipped = np.clip(x, 0.0, 1.0)
    if x_clipped <= a or x_clipped >= c:
        return 0.0
    elif a < x_clipped <= b:
        return (x_clipped - a) / (b - a) if b != a else 1.0
    else:
        return (c - x_clipped) / (c - b) if c != b else 1.0

def fuzzify(feature_val, feature_name):
    """
    Returns a dictionary of membership degrees for LOW, MEDIUM, HIGH
    for a given feature value and feature name.
    """
    if feature_name not in MF_PARAMS:
        raise ValueError(f"Feature name {feature_name} not found in MF_PARAMS.")
        
    params = MF_PARAMS[feature_name]
    return {
        'LOW':    triangular_mf(feature_val, *params['LOW']),
        'MEDIUM': triangular_mf(feature_val, *params['MEDIUM']),
        'HIGH':   triangular_mf(feature_val, *params['HIGH'])
    }

def apply_fuzzy_rules(fuzz_values):
    """
    Apply Mamdani fuzzy rules using the AND-minimum operator.
    Rules:
      R1:  IF HR is HIGH   AND Speed is HIGH    -> Intensity is HIGH
      R2:  IF HR is MEDIUM AND Speed is MEDIUM  -> Intensity is MEDIUM
      R3:  IF HR is LOW    AND Speed is LOW     -> Intensity is LOW
      R4:  IF HR is HIGH   AND Speed is MEDIUM  -> Intensity is MEDIUM
      R5:  IF HR is MEDIUM AND Speed is HIGH    -> Intensity is HIGH
      R6:  IF Cadence is HIGH AND Gait is HIGH  -> Intensity is HIGH
      R7:  IF StepCount is LOW AND Speed is LOW -> Intensity is LOW
      R8*: IF Calories is HIGH AND HR is HIGH   -> Intensity is HIGH
      R9*: IF Calories is LOW  AND Speed is LOW -> Intensity is LOW
      R10*:IF Calories is MEDIUM                -> Intensity is MEDIUM
    """
    hr   = fuzz_values['heart_rate']
    spd  = fuzz_values['speed']
    cad  = fuzz_values['cadence']
    gait = fuzz_values['gait']
    stp  = fuzz_values['step_count']
    cal  = fuzz_values['calories']

    rule_outputs = {
        'HIGH':   max(
                    min(hr['HIGH'],   spd['HIGH']),    # Rule 1
                    min(hr['MEDIUM'], spd['HIGH']),    # Rule 5
                    min(cad['HIGH'],  gait['HIGH']),   # Rule 6
                    min(cal['HIGH'],  hr['HIGH'])      # Rule 8*
                  ),
        'MEDIUM': max(
                    min(hr['MEDIUM'], spd['MEDIUM']),  # Rule 2
                    min(hr['HIGH'],   spd['MEDIUM']),  # Rule 4
                    cal['MEDIUM']                      # Rule 10*
                  ),
        'LOW':    max(
                    min(hr['LOW'],    spd['LOW']),     # Rule 3
                    min(stp['LOW'],   spd['LOW']),     # Rule 7
                    min(cal['LOW'],   spd['LOW'])      # Rule 9*
                  )
    }
    return rule_outputs

def defuzzify_centroid(rule_outputs):
    """
    Centroid defuzzification.
    Formula: crisp_output = sum(weight * centre) / sum(weight)
    Returns a crisp value in [0, 1] and a corresponding baseline label.
    """
    numerator = sum(rule_outputs[lvl] * OUTPUT_CENTRES[lvl] for lvl in rule_outputs)
    denominator = sum(rule_outputs[lvl] for lvl in rule_outputs)

    if denominator == 0:
        return 0.5, 'MEDIUM'  # default

    crisp = numerator / denominator

    if crisp < 0.35:
        label = 'LOW'
    elif crisp < 0.65:
        label = 'MEDIUM'
    else:
        label = 'HIGH'

    return crisp, label
