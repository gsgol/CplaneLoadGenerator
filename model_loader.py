import random
import torch
from typing import List, Dict, Set, Tuple
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils import creates_forbidden_ngram


def load_lm(
    model_name: str = "gpt2",
    device: str = "cpu",
):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device)
    model.eval()
    return tokenizer, model


def prepare_tokenizer(tokenizer, model, proc_list):
    new_tokens = [name for name in proc_list if name not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

    proc2tid = {nm: tokenizer.convert_tokens_to_ids(nm) for nm in proc_list}
    tid2proc = {tid: nm for nm, tid in proc2tid.items()}

    return proc2tid, tid2proc


def sample_next_token(
    model,
    tokenizer,
    seq_ids: List[int],
    candidates: List[str],
    proc2tid: Dict[str, int],
    used_ngrams: Set[Tuple[int, ...]],
    max_ngram: int,
    top_k: int = 20,
) -> int:
    device = next(model.parameters()).device

    cand_ids = torch.tensor(
        [proc2tid[p] for p in candidates],
        dtype=torch.long,
        device=device,
    )

    max_len = model.config.n_positions
    if len(seq_ids) >= max_len:
        ctx = seq_ids[-max_len:]
    else:
        pad_id = tokenizer.pad_token_id
        ctx = [pad_id] * (max_len - len(seq_ids)) + seq_ids

    input_ids = torch.tensor([ctx], dtype=torch.long, device=device)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]

    mask = torch.full_like(logits, float("-inf"))
    mask[0, cand_ids] = 0.0
    probs = torch.softmax(logits + mask, dim=-1)

    k = min(top_k, len(candidates))
    _, top_ids = torch.topk(probs, k=k, dim=-1)
    top_ids = top_ids.squeeze(0).cpu().numpy()

    for tid in top_ids:
        if not creates_forbidden_ngram(seq_ids, int(tid), used_ngrams, n=max_ngram):
            return int(tid)

    return random.choice(cand_ids).item()