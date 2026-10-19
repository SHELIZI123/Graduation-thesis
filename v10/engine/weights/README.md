---
pipeline_tag: tabular-regression
library_name: tabpfn
license: other
license_name: priorlabs-1-1
license_link: https://github.com/PriorLabs/TabPFN/blob/49394b053a6759cfe68e90c21a2d51c31b396768/LICENSE
---

# TabPFN v2: A Tabular Foundation Model

TabPFN is a transformer-based foundation model for tabular data that leverages prior-data based learning to achieve strong performance on small tabular regression tasks without requiring task-specific training.

## Installation
```bash
pip install tabpfn
```

## Model Details
- **Developed by:** Prior Labs
- **Model type:** Transformer-based foundation model for tabular data
- **License:**  [Prior Labs License (Apache 2.0 with additional attribution requirement)](https://priorlabs.ai/tabpfn-license/)
- **Paper:** Published in Nature (January 2025)
- **Repository:** [GitHub - priorlabs/tabpfn](https://github.com/priorlabs/tabpfn)

### 📚 Citation

```bibtex
@article{hollmann2025tabpfn,
 title={Accurate predictions on small data with a tabular foundation model},
 author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and
         Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and
         Schirrmeister, Robin Tibor and Hutter, Frank},
 journal={Nature},
 year={2025},
 month={01},
 day={09},
 doi={10.1038/s41586-024-08328-6},
 publisher={Springer Nature},
 url={https://www.nature.com/articles/s41586-024-08328-6},
}
```

## Quick Start

📚 For detailed usage examples and best practices, check out:
- [Interactive Colab Tutorial](https://tinyurl.com/tabpfn-colab-api)


## Technical Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.1
- scikit-learn ≥ 1.0
- Hardware: 16GB+ RAM, CPU (GPU optional)

## Limitations
- Not designed for very large datasets
- Not suitable for non-tabular data formats

## Resources
- **Documentation:** https://priorlabs.ai/docs
- **Source:** https://github.com/priorlabs/tabpfn
- **Paper:** https://www.nature.com/articles/s41586-024-08328-6

### Team
- Noah Hollmann
- Samuel Müller
- Lennart Purucker
- Arjun Krishnakumar
- Max Körfer
- Shi Bin Hoo
- Robin Tibor Schirrmeister
- Frank Hutter
- Eddie Bergman
- Léo Grinsztajn