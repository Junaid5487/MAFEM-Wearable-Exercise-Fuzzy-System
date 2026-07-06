import numpy as np

class StudentProfile:
    """
    Stores student fitness history and adapts thresholds dynamically.
    Formula from paper: Nthre = OThre + K * (Cfl - Ifl)
    """
    def __init__(self, student_id, initial_fitness=0.5, K=0.1):
        self.student_id = student_id
        self.K = K
        self.initial_fitness = initial_fitness
        self.threshold = 0.5   # starting threshold
        self.history = []    # list of past intensity scores
        self.calorie_history = []    # kcal per window
        self.total_calories_burned = 0.0  # cumulative kcal

        # HR zones (bpm) - will be updated adaptively
        self.hr_zones = {
            'LOW':    (60, 100),
            'MEDIUM': (100, 140),
            'HIGH':   (140, 180)
        }

    def update_threshold(self):
        """Update threshold based on recent session history."""
        if len(self.history) < 3:
            return  # not enough data yet
        current_fl = np.mean(self.history[-5:])  # last 5 sessions
        self.threshold = self.threshold + self.K * (current_fl - self.initial_fitness)
        self.threshold = np.clip(self.threshold, 0.1, 0.9)

        # Update HR zones based on shift
        shift = int((current_fl - self.initial_fitness) * 20)
        self.hr_zones = {
            'LOW':    (60 + shift, 100 + shift),
            'MEDIUM': (100 + shift, 140 + shift),
            'HIGH':   (140 + shift, 180 + shift)
        }

    def add_session(self, intensity_score, calories_kcal=0.0):
        """Record a window's intensity and calories."""
        self.history.append(intensity_score)
        self.calorie_history.append(calories_kcal)
        self.total_calories_burned += calories_kcal
        self.update_threshold()

    def get_calorie_summary(self):
        """Return a dict with calorie statistics for the session."""
        if not self.calorie_history:
            return {}
        arr = np.array(self.calorie_history)
        return {
            'total_kcal':   round(self.total_calories_burned, 2),
            'mean_kcal_per_window': round(float(np.mean(arr)), 4),
            'peak_kcal_window':     round(float(np.max(arr)),  4),
        }

    def get_recommendation(self, intensity_label):
        recs = {
            'LOW':    'Increase pace or duration - current activity is below your target zone.',
            'MEDIUM': 'Good effort! Maintain current intensity to meet your fitness goal.',
            'HIGH':   'Excellent intensity! Ensure adequate recovery time after this session.'
        }
        return recs.get(intensity_label, 'Keep going!')

    def summary(self):
        avg_intensity = np.mean(self.history) if self.history else 0.0
        cal = self.get_calorie_summary()
        
        info = {
            "student_id": self.student_id,
            "threshold": round(self.threshold, 3),
            "sessions_logged": len(self.history),
            "avg_intensity": round(avg_intensity, 3),
            "hr_zones": self.hr_zones,
            "calorie_summary": cal
        }
        return info
