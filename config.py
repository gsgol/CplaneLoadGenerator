from typing import Dict, Set, Tuple

MAX_NGRAM = 1000

NUM_DEVICES = 50

SIMULATION_MODE = "stable"

MODE_PARAMS = {
    "stable": {},
    "one_peak": {
        "peak_position": 0.5,
        "peak_multiplier": 3.0,
        "base_multiplier": 0.3,
        "peak_width": 0.2,
    },
    "sinusoid": {
        "num_periods": 3,
        "amplitude": 0.7,
        "phase_shift": 0.0,
    },
}

PROC_LIST: Dict[str, Tuple[int, int]] = {
    "att": (30, 50),
    "det": (70, 90),
    "im_exit": (25, 30),
    "im_enter": (20, 40),
    "erab_set": (6, 15),
    "erab_rel": (6, 10),
    "ho_in": (50, 74),
    "ho_out": (63, 84),
}

ALLOWED_NEXT: Dict[str, Set[str]] = {
    "att": {"det", "im_enter", "erab_set", "ho_out"},
    "det": {"att", "ho_in"},
    "im_exit": {"det", "im_enter", "erab_set", "ho_out"},
    "im_enter": {"im_exit", "det"},
    "erab_set": {"erab_rel", "det", "im_enter", "ho_out"},
    "erab_rel": {"erab_set", "det", "im_enter", "ho_out"},
    "ho_in": {"det", "im_enter", "erab_set", "ho_out"},
    "ho_out": {"att", "ho_in"},
}

DEFAULT_RATIO: Dict[str, float] = {
    "att": 20.0,
    "det": 20.0,
    "im_exit": 5.0,
    "im_enter": 5.0,
    "erab_set": 20.0,
    "erab_rel": 20.0,
    "ho_in": 5.0,
    "ho_out": 5.0,
}

DEFAULT_TARGET_RATE = 40.0
DEFAULT_SIMULATION_TIME = 5.0
DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_TOP_K = 6
DEFAULT_SEED = 42

DEFAULT_INITIAL_SEQUENCE = [
    "att", "erab_set", "erab_rel", "im_enter",
    "im_exit", "erab_set", "erab_rel", "ho_out", "att",
]