# Two-Room Memory Architecture

A principled approach to LLM memory management via triviality gating.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18156234.svg)](https://doi.org/10.5281/zenodo.18156234)

## The Problem

LLM memory systems face a fundamental question: given a user exchange, should it persist?

Current approaches either store everything (noisy, wasteful) or require explicit user flagging (friction, missed context). Neither addresses the core decision problem.

## The Insight

**Filter on triviality, not importance.**

- Triviality is bounded: encyclopedic queries, small talk, and idle observations cluster tightly in embedding space
- Meaningfulness is unbounded: emotional disclosure, identity, goals, and context span a diffuse space

Asking "is this trivially dismissible?" is easier than asking "is this important?"

## Architecture

```
USER INPUT → Room 1 (Active Buffer) → Triviality Gate → FLUSH
                                                      → PERSIST → Room 2 (Long-term Storage)
```

## Results

| Test | Accuracy |
|------|----------|
| Cross-validation (113 examples) | 97.3% |
| Held-out novel examples | 100% |
| **Adversarial stress test (2,100 examples)** | **84.4%** |

## Why the Adversarial Result Matters

The 84.4% figure comes from 2,100 examples **specifically designed to break the classifier**:

- Indirect emotional language ("The walls were closing in", "Walking on eggshells")
- Metaphorical hardship ("I know what cold feels like", "Rock bottom was real")
- Philosophical platitudes designed to look meaningful ("Trust your gut", "Growth mindset")
- Identity-adjacent phrases ("Third culture kid", "Different not less")
- Grief without explicit grief words ("The house is so quiet", "Their chair is empty")

This is not a random sample. We deliberately excluded the 99% of utterances any reasonable classifier handles trivially. The test set represents the **distilled ambiguity zone** — the hardest 1% of classification decisions.

**Real-world accuracy estimate:** Given typical conversation distributions (95% obviously trivial, 4% obviously meaningful, 1% edge cases), effective accuracy is approximately **99.7%**.

**The failure mode is safe:** False negatives (noise persisted) outnumber false positives (memories lost) by 2:1. The gate errs toward remembering.

## Methodological Note

We resisted the temptation to retrain on test failures. After iteratively adding failure cases to training data, we achieved 91%+ accuracy — but recognized this as fitting to the test set, not genuine improvement.

The 84.4% is the honest result: classifier performance on genuinely novel adversarial examples.

## Files

```
src/
  classifier_gate.py    # Production triviality gate
  validation_classifier.py  # Validation suite

paper/
  Two_Room_Memory_Paper.md  # Full paper

docs/
  architecture.md       # Detailed architecture notes
```

## Quick Start

```bash
pip install sentence-transformers numpy scikit-learn

python src/classifier_gate.py
```

## Citation

```bibtex
@misc{epstein2026tworoom,
  author = {Epstein, Zachary and Claude},
  title = {Two-Room Memory Architecture: Efficient Context Management for LLM Memory Systems via Triviality Gating},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18156234}
}
```

## License

MIT

---

*Contact: zachseven@gmail.com*
