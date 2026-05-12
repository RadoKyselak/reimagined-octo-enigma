# Neuro-symbolic model based on First-Order Logic (FOL)

This repository contains a compact PyTorch implementation of a neuro-symbolic model:

- **Neural part:** embedding-based link prediction (TransE-style scoring).
- **Symbolic part:** FOL implication rules as a differentiable regularizer.

## Core idea

For triples `(predicate, subject, object)` the model learns embeddings and a score:

`score = -||e_subject + r_predicate - e_object||`

Truth is mapped by a sigmoid. FOL rules of the form:

`body(x, y) -> head(x, y)`

are enforced by penalizing violations:

`max(0, truth(body) - truth(head))`

The final objective is:

`data_loss + rule_weight * rule_loss`

## Files

- `neurosymbolic_fol.py`: model, rule loss, demo KB, and training loop.

## Run

```bash
python neurosymbolic_fol.py
```

Expected output shows loss decreasing after training.
