import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter


class SequenceDataLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.sequences = []
        self.metadata = {}

    def load(self) -> List[List[str]]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Файл {self.file_path} не найден")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.metadata = data.get('metadata', {})
            self.sequences = data.get('sequences', [])
        elif isinstance(data, list):
            self.sequences = data
        else:
            raise ValueError("Неверный формат JSON файла")

        return self.sequences

    def get_statistics(self) -> Dict[str, Any]:
        if not self.sequences:
            return {}

        all_procedures = [proc for seq in self.sequences for proc in seq]

        stats = {
            'total_sequences': len(self.sequences),
            'min_length': min(len(seq) for seq in self.sequences),
            'max_length': max(len(seq) for seq in self.sequences),
            'avg_length': sum(len(seq) for seq in self.sequences) / len(self.sequences),
            'total_procedures': len(all_procedures),
            'unique_procedures': len(set(all_procedures)),
            'procedure_counts': dict(Counter(all_procedures)),
        }

        return stats

    def validate_sequences(
        self,
        allowed_procedures: Optional[set] = None,
        allowed_next: Optional[Dict[str, set]] = None,
    ) -> Dict[str, Any]:
        validation_results = {
            'valid_sequences': 0,
            'invalid_sequences': 0,
            'invalid_procedures': [],
            'invalid_transitions': [],
        }

        for idx, seq in enumerate(self.sequences):
            is_valid = True

            if allowed_procedures:
                for proc in seq:
                    if proc not in allowed_procedures:
                        validation_results['invalid_procedures'].append({
                            'sequence_index': idx,
                            'procedure': proc,
                        })
                        is_valid = False

            if allowed_next:
                for i in range(len(seq) - 1):
                    current = seq[i]
                    next_proc = seq[i + 1]

                    if current in allowed_next:
                        if next_proc not in allowed_next[current]:
                            validation_results['invalid_transitions'].append({
                                'sequence_index': idx,
                                'position': i,
                                'from': current,
                                'to': next_proc,
                            })
                            is_valid = False

            if is_valid:
                validation_results['valid_sequences'] += 1
            else:
                validation_results['invalid_sequences'] += 1

        return validation_results

    def filter_valid_sequences(
        self,
        allowed_procedures: Optional[set] = None,
        allowed_next: Optional[Dict[str, set]] = None,
    ) -> List[List[str]]:
        valid_sequences = []

        for seq in self.sequences:
            is_valid = True

            if allowed_procedures:
                if not all(proc in allowed_procedures for proc in seq):
                    is_valid = False

            if allowed_next and is_valid:
                for i in range(len(seq) - 1):
                    current = seq[i]
                    next_proc = seq[i + 1]

                    if current not in allowed_next or next_proc not in allowed_next[current]:
                        is_valid = False
                        break

            if is_valid:
                valid_sequences.append(seq)

        return valid_sequences

    def split_train_test(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[List[List[str]], List[List[str]]]:
        import random
        random.seed(seed)

        shuffled = self.sequences.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_data = shuffled[:split_idx]
        test_data = shuffled[split_idx:]

        return train_data, test_data


def load_multiple_files(file_paths: List[str]) -> List[List[str]]:
    all_sequences = []
    for file_path in file_paths:
        loader = SequenceDataLoader(file_path)
        sequences = loader.load()
        all_sequences.extend(sequences)
    return all_sequences


def save_sequences(
    sequences: List[List[str]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": metadata or {},
        "sequences": sequences,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)