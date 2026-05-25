import time
import json
import torch
from pathlib import Path
from collections import Counter
from typing import Dict

from config import (
    PROC_LIST,
    ALLOWED_NEXT,
    DEFAULT_RATIO,
    DEFAULT_TARGET_RATE,
    DEFAULT_SIMULATION_TIME,
    DEFAULT_MODEL_NAME,
    DEFAULT_TOP_K,
    DEFAULT_SEED,
    DEFAULT_INITIAL_SEQUENCE,
    MAX_NGRAM,
    NUM_DEVICES,
    SIMULATION_MODE,
    MODE_PARAMS,
)
from model_loader import load_lm
from generator import generate_timeline


def print_device_statistics(result: Dict):
    device_stats = result["device_statistics"]
    num_devices = result["num_devices"]
    target_rate = result["target_rate"]
    
    total_procedures = 0
    for dev_id in sorted(device_stats.keys()):
        stats = device_stats[dev_id]
        total_procedures += stats["total_procedures"]
    
    print(f"\n{'Всего':<6} {total_procedures:<12}")
    
    target_rate_per_device = target_rate / num_devices
    print(f"Целевая частота на устройство: {target_rate_per_device:.2f} proc/s")
    print(f"Общая целевая частота: {target_rate:.2f} proc/s")
    print(f"Общая фактическая частота: {total_procedures / result['simulation_time']:.2f} proc/s")


def print_overall_statistics(result: Dict):
    merged = result["merged_timeline"]
    simulation_time = result["simulation_time"]
    target_rate = result["target_rate"]
    num_devices = result["num_devices"]
    mode = result["simulation_mode"]
    
    cnt = Counter(e["procedure"] for e in merged)
    total = sum(cnt.values())
    
    print(f"\n=== Общая статистика ({num_devices} устройств, режим: {mode}) ===")
    for proc in sorted(PROC_LIST):
        print(
            f"{proc:12}: {cnt.get(proc, 0):7d} шт.  "
            f"({cnt.get(proc, 0) / total * 100 if total > 0 else 0:5.2f} %)"
        )
    
    print(f"\nВсего процедур       : {total}")
    print(f"Средняя частота      : {total / simulation_time:.2f} proc/s (цель {target_rate:.2f})")
    print(f"Кол-во устройств     : {num_devices}")


def save_results(result: Dict, output_dir: str = "./results"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mode_suffix = f"_{result['simulation_mode']}"
    
    merged_path = output_path / f"merged_timeline{mode_suffix}.json"
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(result["merged_timeline"], f, indent=2, ensure_ascii=False)
    
    devices_dir = output_path / f"devices{mode_suffix}"
    devices_dir.mkdir(exist_ok=True)
    for dev_id, timeline in result["device_timelines"].items():
        dev_path = devices_dir / f"device_{dev_id}.json"
        with open(dev_path, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
    
    serializable_stats = {str(dev_id): stats for dev_id, stats in result["device_statistics"].items()}
    
    summary = {
        "num_devices": result["num_devices"],
        "simulation_time": result["simulation_time"],
        "target_rate": result["target_rate"],
        "simulation_mode": result["simulation_mode"],
        "mode_params": result["mode_params"],
        "device_statistics": serializable_stats,
    }
    
    stats_path = output_path / f"statistics{mode_suffix}.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def save_metrics(model_name, total_time, peak_mem, num_devices, mode, log_path="metrics.txt"):
    log_line = (
        f"model={model_name} | "
        f"mode={mode} | "
        f"devices={num_devices} | "
        f"time={total_time:.2f}s | "
        f"peak_mem={peak_mem:.1f}MiB\n"
    )
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(log_line)


def main(
    model_path: str = None,
    num_devices: int = None,
    mode: str = None,
):
    model_to_use = model_path or DEFAULT_MODEL_NAME
    devices = num_devices or NUM_DEVICES
    sim_mode = mode or SIMULATION_MODE
    mode_params = MODE_PARAMS.get(sim_mode, {})

    tokenizer, model = load_lm(model_to_use)
    
    print(f"\n=== Параметры симуляции ===")
    print(f"Режим: {sim_mode}")
    print(f"Параметры режима: {mode_params}")
    print(f"Количество устройств: {devices}")
    print(f"Целевая частота: {DEFAULT_TARGET_RATE} proc/s (суммарно)")
    print(f"Время симуляции: {DEFAULT_SIMULATION_TIME} сек")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    
    result = generate_timeline(
        proc_list=PROC_LIST,
        ratio=DEFAULT_RATIO,
        allowed_next=ALLOWED_NEXT,
        target_rate=DEFAULT_TARGET_RATE,
        simulation_time=DEFAULT_SIMULATION_TIME,
        model=model,
        tokenizer=tokenizer,
        seed=DEFAULT_SEED,
        max_ngram=MAX_NGRAM,
        top_k=DEFAULT_TOP_K,
        initial_seq=DEFAULT_INITIAL_SEQUENCE,
        num_devices=devices,
        simulation_mode_name=sim_mode,
        mode_params=mode_params,
    )
    
    total_time = time.time() - start_time
    peak_mem = 0.0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print_device_statistics(result)
    print_overall_statistics(result)
    
    save_metrics(model_to_use, total_time, peak_mem, devices, sim_mode)
    save_results(result)


if __name__ == "__main__":
    import sys
    
    model_path = None
    num_devices = None
    sim_mode = None
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model_path = args[i + 1]
            i += 2
        elif args[i] == "--devices" and i + 1 < len(args):
            num_devices = int(args[i + 1])
            i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            sim_mode = args[i + 1]
            i += 2
        else:
            model_path = args[i]
            i += 1
    
    main(model_path=model_path, num_devices=num_devices, mode=sim_mode)