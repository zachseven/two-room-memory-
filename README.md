# Two-Room Memory Architecture

**Efficient LLM memory management via triviality gating**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](link-to-arxiv-when-published)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the prototype implementation for the Two-Room Memory Architecture, a novel approach to LLM memory management that filters on triviality rather than importance.

**Key insight:** Triviality forms a tighter semantic cluster than meaningfulness. Instead of asking "is this important enough to remember?" we ask "is this trivially dismissible?"

**Result:** A classifier trained on 113 examples achieves 100% accuracy on novel test cases, demonstrating that a cheap, learnable, generalizable decision boundary exists between trivial and meaningful exchanges.

```
USER INPUT → Room 1 (Active Buffer) → Triviality Gate → FLUSH (trivial)
                                                      → PERSIST → Room 2 (Storage)
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/two-room-memory.git
cd two-room-memory

# Install dependencies
pip install -r requirements.txt

# Run the gate test
python classifier_gate.py

# Run full validation
python validation_classifier.py
```

## Results

| Test Set | Accuracy | False Positives | False Negatives |
|----------|----------|-----------------|-----------------|
| Validation (98 examples) | 100% | 0 | 0 |
| Novel examples (13) | 100% | 0 | 0 |

**Example classifications:**

```
FLUSH:
  0.70 - "what's the square root of 144"
  0.78 - "how tall is the eiffel tower"
  0.63 - "nice weather we're having"

PERSIST → Room 2:
  0.76 - "my grandmother raised me"
  0.81 - "i'm autistic and it affects my relationships"
  0.75 - "i'm going through a rough patch"
```

## Repository Structure

```
two-room-memory/
├── README.md
├── requirements.txt
├── LICENSE
├── paper/
│   └── Two_Room_Memory_Paper.md
├── src/
│   ├── classifier_gate.py      # Main triviality gate (classifier-based)
│   ├── validation_classifier.py # Validation suite
│   ├── room1_gate_neural.py    # Original neural approach
│   └── room1_gate.py           # TF-IDF baseline (for reference)
└── docs/
    └── architecture.md         # Full architecture specification
```

## The Triviality Gate

The gate uses sentence embeddings + logistic regression to classify exchanges:

```python
from src.classifier_gate import process_exchange

result = process_exchange("my dad died yesterday")
# {'exchange': '...', 'decision': 'PERSIST', 'confidence': 0.84}

result = process_exchange("what color are ladybugs")
# {'exchange': '...', 'decision': 'FLUSH', 'confidence': 0.73}
```

## Room 2 Design (Theoretical)

The paper proposes organizing persistent memory by **relational posture** rather than data category:

| Category | Relational Demand | Example |
|----------|-------------------|---------|
| EMPATHY | Emotional attunement | "My mom died yesterday" |
| UNDERSTANDING | Accommodation, patience | "I have ADHD" |
| RESPECT | Recognition of capability | "I went to law school" |
| COMMUNICATION | Behavioral adjustment | "I prefer direct communication" |
| CONTEXT | Useful if relevant | "I work as a contractor" |
| VOLATILE | Important now, may change | "I'm shipping my game in January" |

See the [full paper](paper/Two_Room_Memory_Paper.md) for details on tier assignment and retrieval mechanisms.

## Citation

If you use this work, please cite:

```bibtex
@article{epstein2026tworoom,
  title={Two-Room Memory Architecture: Efficient Context Management for LLM Memory Systems via Triviality Gating},
  author={Epstein, Zachary and Claude},
  year={2026},
  note={arXiv preprint}
}
```
https://doi.org/10.5281/zenodo.18156234

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Zachary Epstein — zachseven@gmail.com

---

*Developed in collaboration with Claude (Anthropic)*
