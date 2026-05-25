import random
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass, field

from simulation_modes import SimulationMode


@dataclass
class DeviceState:
    device_id: int
    cur_name: str = ""
    cur_time: float = 0.0
    seq_ids: List[int] = field(default_factory=list)
    used_ngrams: set = field(default_factory=set)
    remaining: Dict[str, int] = field(default_factory=dict)
    timeline: List[Dict] = field(default_factory=list)
    active: bool = True


class DeviceManager:
    def __init__(
        self,
        num_devices: int,
        proc_list: Dict[str, Tuple[int, int]],
        allowed_next: Dict[str, Set[str]],
        ratio: Dict[str, float],
        target_rate: float,
        simulation_time: float,
        proc2tid: Dict[str, int],
        simulation_mode: SimulationMode,
        initial_seq: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.num_devices = num_devices
        self.proc_list = proc_list
        self.allowed_next = allowed_next
        self.ratio = ratio
        self.target_rate = target_rate
        self.simulation_time = simulation_time
        self.proc2tid = proc2tid
        self.tid2proc = {v: k for k, v in proc2tid.items()}
        self.simulation_mode = simulation_mode
        self.initial_seq = initial_seq
        self.seed = seed
        self.devices: List[DeviceState] = []
    
    def initialize_devices(self):
        from utils import compute_counts, add_ngrams
        
        average_rate = self.simulation_mode.get_average_rate()
        rate_per_device = average_rate / self.num_devices
        total_per_device = int(round(rate_per_device * self.simulation_time * 1.5))
        
        for i in range(self.num_devices):
            random.seed(self.seed + i)
            
            remaining = compute_counts(self.ratio, total_per_device)
            device = DeviceState(device_id=i, remaining=remaining)
            
            if self.initial_seq:
                valid = True
                for n in self.initial_seq:
                    if remaining.get(n, 0) == 0:
                        valid = False
                        break
                
                if valid:
                    for n in self.initial_seq:
                        device.remaining[n] -= 1
                    
                    device.seq_ids = [self.proc2tid[n] for n in self.initial_seq]
                    for j in range(len(device.seq_ids)):
                        add_ngrams(
                            device.seq_ids[:j],
                            device.seq_ids[j],
                            device.used_ngrams,
                            n=1000,
                        )
                    device.cur_name = self.initial_seq[-1]
                else:
                    cur_name = random.choice(list(self.allowed_next))
                    device.cur_name = cur_name
                    device.seq_ids = [self.proc2tid[cur_name]]
                    add_ngrams([], self.proc2tid[cur_name], device.used_ngrams, n=1000)
            else:
                offset = random.uniform(0, 1.0 / max(rate_per_device, 0.001))
                device.cur_time = offset
                cur_name = random.choice(list(self.allowed_next))
                device.cur_name = cur_name
                device.seq_ids = [self.proc2tid[cur_name]]
                add_ngrams([], self.proc2tid[cur_name], device.used_ngrams, n=1000)
            
            self.devices.append(device)
    
    def get_active_devices(self) -> List[DeviceState]:
        return [d for d in self.devices if d.active]
    
    def get_next_device(self) -> Optional[DeviceState]:
        active = self.get_active_devices()
        if not active:
            return None
        return min(active, key=lambda d: d.cur_time)
    
    def deactivate_device(self, device_id: int):
        for d in self.devices:
            if d.device_id == device_id:
                d.active = False
                break
    
    def get_all_timelines(self) -> Dict[int, List[Dict]]:
        return {d.device_id: d.timeline for d in self.devices}
    
    def get_merged_timeline(self) -> List[Dict]:
        merged = []
        for d in self.devices:
            for event in d.timeline:
                merged_event = dict(event)
                merged_event["device_id"] = d.device_id
                merged.append(merged_event)
        merged.sort(key=lambda e: e["start"])
        return merged
    
    def get_device_statistics(self) -> Dict[int, Dict]:
        from collections import Counter
        
        stats = {}
        for d in self.devices:
            cnt = Counter(e["procedure"] for e in d.timeline)
            total = sum(cnt.values())
            stats[d.device_id] = {
                "total_procedures": total,
                "procedure_counts": dict(cnt),
                "actual_rate": total / self.simulation_time if self.simulation_time > 0 else 0,
                "last_event_time": d.timeline[-1]["end"] if d.timeline else 0,
                "active": d.active,
            }
        return stats