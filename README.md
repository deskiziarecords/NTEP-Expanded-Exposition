# NTEP – Expanded Exposition

**Neuro‑symbolic Tool Composition via Spherical Embeddings**  
Author: J. Roberto Jiménez ✉️ `tijuanapaint@gmail.com`  
DOI: <https://doi.org/10.5281/zenodo.18398188>  

This repository contains the **full technical description**, **pre‑trained
embeddings**, and **minimal PyTorch scripts** to reproduce the NTEP framework
presented in the Zenodo article.

---

## 📚 Repository Overview

| Path                     | Description |
|--------------------------|-------------|
| `docs/axioms.pdf`       | Printable version of the *Axiomatic Foundation* section. |
| `scripts/train.py`       | Train spherical embeddings with the contrastive loss (see **A1**). |
| `scripts/infer.py`       | Decode a valid tool chain from a prompt embedding. |
| `data/embeddings.pt`     | Saved embedding matrix after training (`torch.save`). |
| `requirements.txt`       | Python dependencies. |

---

## 🚀 Quick‑Start

```bash
# 1️⃣ Clone the repo
git clone https://github.com/dezkiziarecords/ntep-expanded-exposition.git
cd ntep-expanded-exposition

# 2️⃣ Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Train embeddings on the toy dataset
python scripts/train.py --epochs 10 --batch_size 64

# 5️⃣ Run inference on a synthetic pipeline
python scripts/infer.py --prompt "process document" --chain "resize package email"

```
---
📖 Documentation

The core technical narrative lives in docs/axioms.pdf. It expands on:

    Symbolic definitions (𝒯, ℰ, ≤, γ, δ…)
    Category‑theoretic construction (𝒫, metric enrichment, embedding functor)
    Monotonicity Theorem with full proof sketch
    Consciousness measure built from density‑matrix entropy
    Generalization bounds, geometric properties, and generalization guarantees

### Contributing

Feel free to open issues or pull requests if you spot a typo, need a
visualisation, or want to extend the framework (e.g., stochastic composition,
higher‑order interactions).

---

### 📜 Citation

If you use NTEP in research, please cite the Zenodo record:

``` bibitex
@misc{jimenez2023ntep,
  author       = {Jiménez, J. Roberto},
  title        = {{NTEP}: Expanded Exposition},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18398188},
  url          = {https://doi.org/10.5281/zenodo.18398188}
}
```
📧 Contact

For questions or collaboration proposals, reach out to J. Roberto Jiménez
at tijuanapaint@gmail.com.

-----
 If you would like to support the development of these resources, consider contributing towards helping me get some gear for continued improvement or simply treating me to a coffee. Your support means a lot!" 
 [buy me a coffee](buymeacoffee.com/hipotermiah)
 
