# Using LOAM outside the benchmark

The LOAM models (`Soilytix/LOAM-25M`, `-100M`, `-340M`, `-624M`) are ordinary Hugging Face
causal language models over single nucleotides. They ship their own modeling code, so they load
with `trust_remote_code=True`. For general use (scoring, embeddings, generation, input
preparation), follow the model card on the Hub. This page covers what the card does not: how
the benchmark reads the models, so that you can reproduce its numbers in your own code.

The Hub repositories are gated: accept the terms on the model page, then log in
(`hf auth login`).

## How the benchmark differs from the model card

| | Model card (general use) | gLMBench (the paper's numbers) |
|---|---|---|
| Token before the sequence | none: the tokenizer adds none, and LOAM was trained with no token before a sequence | a `BOS` token, as context only (never scored, never pooled), because the paper's scoring code prepended one |
| Context | 8,192 nt | 8,191 nt: `BOS` takes one of the 8,192 positions |
| Embedding of layer *i* | `hidden_states[i]`; the card's example pools the last one, the final-normed state | the raw output of block *i*, read with a forward hook; the final RMSNorm is never applied |
| Precision | fp32 | weights loaded in fp32, then cast to bf16 on the GPU |

Neither column is wrong; they answer different questions. Do not mix them: a number computed one
way is not comparable with the paper's numbers, which were computed the other way.

How much each difference matters:

- **The `BOS` prefix** barely moves likelihoods but does move embeddings. Measured on LOAM-25M,
  on 126 genes it was not trained on: the mean negative log-likelihood of nucleotides 2 to N
  was 1.24988 nats with no prefix and 1.25008 with `BOS`, while mean-pooled block outputs moved
  by up to 4% (last block). Small, but reproducing the paper exactly needs the `BOS`. `BOS`
  itself is never a prediction target, but the benchmark does score the first nucleotide given
  `BOS`, as the paper's scoring code did. That nucleotide scored 2.16 nats, worse than a uniform
  guess over four bases (1.39 nats): the model never learned `BOS` as a start.
- **The final norm** changes what an embedding is. LOAM's residual stream carries a few very
  large dimensions, and a per-token RMSNorm divides every other dimension by them, so pooled
  final-normed states and pooled raw block outputs are different features. Log-probabilities
  *do* go through the final norm: that is the model's own forward.

## Reproducing a benchmark score and embedding in your own code

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Soilytix/LOAM-25M"
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, dtype=torch.float32)
model = model.to("cuda", torch.bfloat16).eval()          # load in fp32, then cast (see below)
cfg = model.config

seq = "ATGAAAAAGCTGCTGGCAATTGCCGGTCTGGCCGCTGGCGCAGCCGCACAGGCAGAAGAAACCTGA"
ids = [cfg.bos_token_id] + tokenizer(seq, add_special_tokens=False)["input_ids"]
ids = torch.tensor([ids[: cfg.max_position_embeddings]], device="cuda")

# Zero-shot score: the mean log-probability of each nucleotide given BOS and what precedes it.
with torch.inference_mode():
    logits = model(input_ids=ids).logits
logp = torch.log_softmax(logits[0, :-1].float(), dim=-1).gather(-1, ids[0, 1:, None]).squeeze(-1)
print(logp.mean().item())                                   # nats per nucleotide

# Embeddings: layer 0 is the token-embedding output, layer i the raw output of block i.
taps = []
hooks = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: taps.append(o))]
hooks += [b.register_forward_hook(lambda m, i, o: taps.append(o)) for b in model.model.layers]
with torch.inference_mode():
    model.model(input_ids=ids)
for h in hooks:
    h.remove()
per_layer = torch.stack(taps)[:, 0, 1:].float().mean(dim=1)   # [layers, hidden]: drop BOS, mean-pool
```

## Batches

Right-pad after `[BOS, ...]` and run without an attention mask. Under the causal mask a real
token only attends to positions at or before it, and every pad sits after every real token,
so the real positions are exact. Exclude the pads (and BOS) when you pool or score. Left
padding needs an attention mask; the model supports one, but it is not the path the
benchmark's numbers were produced on.

## Getting the paper's numbers exactly

The benchmark's LOAM rows reproduce the paper bitwise because the `loam-hf` runner matches the
arithmetic of the implementation the paper was scored with, not only its semantics:

- weights loaded in fp32, then cast to bf16 on the GPU;
- right-padded `[BOS, ...]` batches through `model.model` with no attention mask;
- embedding and log-prob batches bucketed by length under a 16,384-token budget, and sequence
  scores in length-sorted chunks of 8;
- mean pooling in bf16; log-softmax in fp32.

Loading directly in bf16, pooling in fp32 or batching differently gives numbers that agree to
within bf16 noise (per token, a mean of about 0.01 nats between bf16 and fp32), not bitwise.
For your own work the more accurate choices (fp32 pooling, `pool_dtype: float32` in a spec) are
fine; they change the model hash, so such a run is a different row from the paper's.

For generation, fine-tuning and the models' licence and intended use, see the model cards on
the Hub.
