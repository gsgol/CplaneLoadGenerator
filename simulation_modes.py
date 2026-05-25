import math
from typing import Dict, Callable


class SimulationMode:
    def __init__(self, mode_name: str, params: Dict, simulation_time: float, base_rate: float):
        self.mode_name = mode_name
        self.params = params
        self.simulation_time = simulation_time
        self.base_rate = base_rate
    
    def get_rate_at(self, t: float) -> float:
        raise NotImplementedError
    
    def get_average_rate(self) -> float:
        num_samples = 1000
        rates = [self.get_rate_at(self.simulation_time * i / num_samples) for i in range(num_samples)]
        return sum(rates) / len(rates)


class StableMode(SimulationMode):
    def get_rate_at(self, t: float) -> float:
        return self.base_rate


class OnePeakMode(SimulationMode):
    def get_rate_at(self, t: float) -> float:
        peak_position = self.params.get("peak_position", 0.5)
        peak_multiplier = self.params.get("peak_multiplier", 3.0)
        base_multiplier = self.params.get("base_multiplier", 0.3)
        peak_width = self.params.get("peak_width", 0.2)
        
        normalized_t = t / self.simulation_time
        distance = abs(normalized_t - peak_position)
        
        sigma = peak_width / 2.0
        gauss = math.exp(-(distance ** 2) / (2 * sigma ** 2))
        
        multiplier = base_multiplier + (peak_multiplier - base_multiplier) * gauss
        return self.base_rate * multiplier


class SinusoidMode(SimulationMode):
    def get_rate_at(self, t: float) -> float:
        num_periods = self.params.get("num_periods", 3)
        amplitude = self.params.get("amplitude", 0.7)
        phase_shift = self.params.get("phase_shift", 0.0)
        
        normalized_t = t / self.simulation_time
        sine_value = math.sin(2 * math.pi * num_periods * normalized_t + phase_shift)
        
        multiplier = 1.0 + amplitude * sine_value
        multiplier = max(0.1, multiplier)
        
        return self.base_rate * multiplier


def create_simulation_mode(
    mode_name: str,
    params: Dict,
    simulation_time: float,
    base_rate: float,
) -> SimulationMode:
    modes = {
        "stable": StableMode,
        "one_peak": OnePeakMode,
        "sinusoid": SinusoidMode,
    }
    
    if mode_name not in modes:
        raise ValueError(f"{mode_name} не входит в доступные режимы: {list(modes.keys())}")
    
    return modes[mode_name](mode_name, params, simulation_time, base_rate)


def normalize_mode_to_target_rate(
    mode: SimulationMode,
    target_average_rate: float,
) -> SimulationMode:
    current_average = mode.get_average_rate()
    if current_average <= 0:
        return mode
    
    scale_factor = target_average_rate / current_average
    mode.base_rate = mode.base_rate * scale_factor
    return mode