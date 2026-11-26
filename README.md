# ExPoseFit

Research implementation for RGB exercise recognition, repetition counting and rule-based movement feedback.

## Data

The fitness clips originate from Kinetics-700-2020. The CSV manifests in `data/` identify the custom classification partitions (2,700 entries), the repetition-counting cohort (105 clips), and the integrated evaluation cohort (34 clips). The original classification manifest is retained, including its duplicate training entry.

- [Original 105 videos](https://drive.google.com/drive/folders/1USRz8Wf9mhLry81P1nhx6VeCO9_DlZw0?usp=sharing)
- [System-annotated videos](https://drive.google.com/drive/folders/1OZnVxt7rrXXfPKxPyYfqto1sd9wL9Z9H?usp=sharing)

Video access is subject to original rights and availability. The annotated videos are system outputs, not ground-truth labels.

## Code

`code/` contains the original assessment, MoveNet and utility modules, TSN training notebook and configuration, angle export and online pose benchmark notebooks. The notebooks/configuration retain original Colab/Kaggle paths; adjust them to your local data and code directories. Install MMAction2 and compatible PyTorch/MMCV/MMEngine separately following the [official instructions](https://mmaction2.readthedocs.io/). Exact historical dependency versions have not been verified. Model weights are not included.

## Experiments

Install the numerical dependencies (`pip install numpy pandas scipy`). To reproduce the offline threshold comparison from cached smoothed angles, run from the repository root:

```bash
python experiments/threshold_cv_verified.py
```

This script uses the corrected ground-truth manifest and writes result CSVs into `experiments/`. The other scripts require the original per-frame raw angle trace CSVs (not included in this draft). Run each with `--help` for arguments. `experiments/smoothing_reference.csv` contains the corrected reference predictions used by the updating and frame-rate scripts.

