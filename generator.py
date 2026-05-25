import random
import torch
from typing import Dict, Set, List, Tuple, Optional

from utils import creates_forbidden_ngram, add_ngrams
from model_loader import sample_next_token, prepare_tokenizer
from device import DeviceManager, DeviceState
from simulation_modes import create_simulation_mode, normalize_mode_to_target_rate


def compute_duration_scale(
    cur_time: float,
    simulation_mode,
    base_rate: float,
) -> float:
    current_rate = simulation_mode.get_rate_at(cur_time)
    if current_rate <= 0:
        return 10.0
    return base_rate / current_rate


def step_device(
    device: DeviceState,
    proc_list: Dict[str, Tuple[int, int]],
    allowed_next: Dict[str, Set[str]],
    simulation_time: float,
    model,
    tokenizer,
    proc2tid: Dict[str, int],
    tid2proc: Dict[int, str],
    max_ngram: int,
    top_k: int,
    simulation_mode,
    base_rate: float,
) -> bool:
    if device.cur_time >= simulation_time:
        device.active = False
        return False
    
    if device.remaining.get(device.cur_name, 0) == 0:
        device.active = False
        return False
    
    device.remaining[device.cur_name] -= 1
    
    candidates = [
        p for p in allowed_next.get(device.cur_name, set())
        if device.remaining.get(p, 0) > 0
        and not creates_forbidden_ngram(
            device.seq_ids, proc2tid[p], device.used_ngrams, n=max_ngram
        )
    ]
    
    if not candidates:
        device.active = False
        return False
    
    next_tid = sample_next_token(
        model=model,
        tokenizer=tokenizer,
        seq_ids=device.seq_ids,
        candidates=candidates,
        proc2tid=proc2tid,
        used_ngrams=device.used_ngrams,
        max_ngram=max_ngram,
        top_k=top_k,
    )
    
    proc_name = tid2proc[next_tid]
    dur_ms = random.uniform(*proc_list[proc_name])
    dur_sec = dur_ms / 1000.0
    
    duration_scale = compute_duration_scale(device.cur_time, simulation_mode, base_rate)
    dur_sec *= duration_scale
    
    event = {
        "procedure": proc_name,
        "start": device.cur_time,
        "duration": dur_sec,
        "end": device.cur_time + dur_sec,
    }
    
    device.timeline.append(event)
    device.cur_time += dur_sec
    device.seq_ids.append(next_tid)
    add_ngrams(device.seq_ids[:-1], next_tid, device.used_ngrams, n=max_ngram)
    device.cur_name = proc_name
    
    return True


def generate_timeline(
    proc_list: Dict[str, Tuple[int, int]],
    ratio: Dict[str, float],
    allowed_next: Dict[str, Set[str]],
    target_rate: float,
    simulation_time: float,
    model,
    tokenizer,
    seed: int = None,
    max_ngram: int = 1000,
    top_k: int = 20,
    initial_seq: Optional[List[str]] = None,
    num_devices: int = 1,
    simulation_mode_name: str = "stable",
    mode_params: Optional[Dict] = None,
) -> Dict:
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    proc2tid, tid2proc = prepare_tokenizer(tokenizer, model, proc_list)
    
    simulation_mode = create_simulation_mode(
        mode_name=simulation_mode_name,
        params=mode_params or {},
        simulation_time=simulation_time,
        base_rate=target_rate,
    )
    
    simulation_mode = normalize_mode_to_target_rate(simulation_mode, target_rate)
    
    manager = DeviceManager(
        num_devices=num_devices,
        proc_list=proc_list,
        allowed_next=allowed_next,
        ratio=ratio,
        target_rate=target_rate,
        simulation_time=simulation_time,
        proc2tid=proc2tid,
        simulation_mode=simulation_mode,
        initial_seq=initial_seq,
        seed=seed or 42,
    )
    
    manager.initialize_devices()
    
    while True:
        device = manager.get_next_device()
        if device is None:
            break
        
        success = step_device(
            device=device,
            proc_list=proc_list,
            allowed_next=allowed_next,
            simulation_time=simulation_time,
            model=model,
            tokenizer=tokenizer,
            proc2tid=proc2tid,
            tid2proc=tid2proc,
            max_ngram=max_ngram,
            top_k=top_k,
            simulation_mode=simulation_mode,
            base_rate=target_rate,
        )
        
        if not success:
            manager.deactivate_device(device.device_id)
    
    for device in manager.devices:
        if device.timeline and device.timeline[-1]["end"] > simulation_time:
            device.timeline[-1]["end"] = simulation_time
    
    result = {
        "device_timelines": manager.get_all_timelines(),
        "merged_timeline": manager.get_merged_timeline(),
        "device_statistics": manager.get_device_statistics(),
        "num_devices": num_devices,
        "simulation_time": simulation_time,
        "target_rate": target_rate,
        "simulation_mode": simulation_mode_name,
        "mode_params": mode_params or {},
    }
    
    return result