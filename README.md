# Distilling Wide ResNet Teachers into a Compact Pre-Activation ResNet for CIFAR-10

Code for our NYU Tandon Deep Learning (Fall 2026) mini-project, Kaggle competition
[NYU Tandon Deep Learning F26](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl)
(`cifar-10-res-net-architecture-challenge-dl`).

**Kaggle team:** AI don't kill me plz &nbsp;·&nbsp; **Members:** Minghao Wang, Ningning Han, Xinyue Chen

Our final model is a pre-activation ResNet with **4,696,394 trainable parameters** (limit: 5,000,000), trained from
scratch on the 50,000 CIFAR-10 training images provided for the competition. During training it is distilled from a
WRN-28-10 teacher that was itself trained from scratch on the same images; the teacher is not part of the submitted
model.

| Final model | Notebook | Trainable parameters | Public LB | Private LB |
|---|---|---|---|---|
| Step 13 | [`step13_kd_long_full.ipynb`](step13_kd_long_full.ipynb) | 4,696,394 | 0.918 | **0.881** (12th of 81 teams) |

Our two selected final submissions were Step 13 and Step 12 (public 0.910, private 0.867).

## Where each repository requirement is covered

Competition rule 13 lists what the repository must contain:

| Requirement | Where to find it |
|---|---|
| Model architecture implementation | Section 5 of every notebook (`PreActBlock`, `PreActResNet`); `py_scripts/src/model.py`; [Model architecture](#model-architecture) |
| Training and evaluation code | Sections 6–8 of every notebook (`train` and `predict`; the distillation steps add `teacher_probs`, `kd_loss` and `select_teachers`); `py_scripts/train.py`, `py_scripts/predict.py` |
| Data preprocessing and augmentation | Sections 3–4 of every notebook (`prepare_data`, `stratified_split`, `build_transforms`, `Cutout`) and, from Step 3, `cutmix` in Section 6; `py_scripts/prepare_data.py`, `py_scripts/src/data.py`; [Data preprocessing and augmentation](#data-preprocessing-and-augmentation) |
| Training configuration (optimizer, learning rate, batch size, epochs) | `CONFIG` cell (Section 2) of every notebook, saved with each run as `config.json` (`config.yaml` for Steps 1–2); [Training configuration](#training-configuration), [Experiments and results](#experiments-and-results) |
| Final model parameter count | 4,696,394: printed in Section 5 of every notebook, checked against 5,000,000 before training, and stored in `run_info.json` / `eval.json`; [Model architecture](#model-architecture) |
| Instructions for reproducing the submitted predictions | [Reproducing the submitted predictions](#reproducing-the-submitted-predictions) |
| Citations | [References](#references) |

The configuration, per-epoch training log, validation result and submission file of every run are in
[`runs/`](runs) ([Run records](#run-records)).

## Repository structure

```
├── step1_baseline.ipynb … step13_kd_long_full.ipynb   one self-contained notebook per experiment step
├── step6_final_hpc_output.ipynb    executed Step 6a run (step6_final.ipynb at that time, with its output)
├── runs/
│   └── step<N>_<name>/             record of each run: config.json, run_info.json, log.csv, eval.json, submission.csv
├── py_scripts/                     command-line version of Steps 1–5 (used to run Steps 1 and 2)
│   ├── prepare_data.py  train.py  predict.py  count_params.py
│   ├── configs/                    baseline.yaml, main.yaml, main_cutmix.yaml, teacher_wrn28_10.yaml, kd.yaml
│   ├── src/                        data.py, model.py, utils.py
│   ├── run_step{1,2,3}_local.bat   Windows helpers for a local conda environment
│   └── requirements.txt
├── requirements.txt
└── .gitignore                      excludes data/ and checkpoints (*.pt)
```

Every notebook contains all the code it runs, in the same eight sections: 1 imports and settings, 2 `CONFIG`,
3 data preparation, 4 dataset, split and augmentation, 5 model, 6 training utilities, 7 training and prediction
functions, 8 run. Code that a step adds to the previous one is marked `added in Step N`. The notebooks of Steps 3, 3b,
4, 7, 8 and 9 are saved with the output of their runs.

## Model architecture

Pre-activation ResNet [1, 2] with three stages (`PreActResNet` in Section 5 of each notebook):

| Part | Layers | Output | Trainable parameters |
|---|---|---|---|
| Stem | 3×3 conv, 3→64, stride 1, no max-pooling | 64×32×32 | 1,728 |
| Stage 1 | 4 pre-activation basic blocks, 64 channels | 64×32×32 | 295,936 |
| Stage 2 | 4 blocks, 128 channels, first block with stride 2 | 128×16×16 | 1,116,032 |
| Stage 3 | 3 blocks, 256 channels, first block with stride 2 | 256×8×8 | 3,279,616 |
| Head | BN → ReLU → global average pooling → FC 256→10 | 10 | 3,082 |
| **Total** | | | **4,696,394** |

Residual block: y = S(x) + F(x), with F = BN-ReLU-Conv3×3-BN-ReLU-Conv3×3 and S = identity; when a block changes the
resolution or the width, S is 2×2 average pooling followed by a 1×1 convolution, applied to the pre-activated input
(ResNet-D shortcut [4]). Convolutions have no bias and use Kaiming-normal initialization.

The count can be checked without data: `cd py_scripts && python count_params.py configs/main.yaml configs/teacher_wrn28_10.yaml`.

**Teacher.** The same class with widths 160/320/640, 4 blocks per stage and a 16-channel stem is a WRN-28-10 [3] with
36,479,194 parameters. Teachers (Steps 4, 7, 9, 11) are trained like the students and are used only to compute
training targets.

## Data preprocessing and augmentation

- **Input files (Section 3, `prepare_data`).** The competition PNG files are read once into a NumPy cache in
  `data/cache/`: the 50,000 labeled training images and the 2,000 test images, the latter in the order of
  `sample_submission.csv`. The per-channel mean and standard deviation for normalization are computed from the
  training images only.
- **Split (Section 4, `stratified_split`).** Stratified 45,000 / 5,000 train / validation split (500 validation images
  per class, seed 42), identical in every step. Steps 6 and 13 train on all 50,000 images without a validation split.
- **Per-image training augmentation (`build_transforms`).** Random 32×32 crop of the image padded by 4 pixels
  (reflection) → random horizontal flip → TrivialAugmentWide [5] (torchvision implementation [11]) → conversion to a
  tensor and normalization → Cutout [6]: one 16×16 square of the normalized image set to 0, clipped at the border.
  Step 1 uses only the crop and the flip.
- **Batch-level augmentation (`cutmix`, from Step 3).** CutMix [7] is applied to a batch with probability 0.5; the
  mixing ratio is drawn from Beta(1, 1) and recomputed from the clipped box area, and the two labels are weighted by
  this ratio in the loss. In distillation steps the teachers see the same mixed batch.
- **Validation and test images.** Normalization only. Predictions use the EMA weights and average the softmax outputs
  for the image and its horizontal flip (from Step 2).

## Training configuration

Settings of the final model (Step 13). The optimizer, learning rate, batch size, schedule and weight decay are the
same in every step; the number of epochs and the changes of each step are listed under
[Experiments and results](#experiments-and-results).

| Setting | Value |
|---|---|
| Optimizer | SGD with Nesterov momentum 0.9 |
| Learning rate | Peak 0.1; linear warm-up over 5 epochs, then cosine decay to 0, updated every iteration [4] |
| Batch size | 128 (the incomplete last batch of an epoch is dropped) |
| Epochs | 2,000 |
| Weight decay | 5 × 10⁻⁴ on convolution and linear weights; none on BatchNorm parameters and biases [4] |
| Loss | Cross-entropy with label smoothing 0.1 [8]; with CutMix, the two labels are weighted by the mixing ratio |
| Knowledge distillation | Loss = (1 − α)·CE + α·T²·KL(teacher ‖ student), both distributions softened with temperature T; α = 0.9, T = 4 [9]. Teacher: the Step 9 WRN-28-10 (EMA weights, frozen), which sees the same augmented batch as the student [10] |
| Weight averaging | EMA of the weights (BatchNorm statistics included) with decay min(0.999, (1 + n)/(10 + n)) at update n; the EMA weights are evaluated and used for prediction |
| Precision | Mixed precision: bfloat16 autocast on GPUs that support it (the A100), otherwise float16 with gradient scaling |
| Seeds | 42 for the data split and for training (the Step 9 and Step 11 teachers use training seeds 1 and 2) |
| Parameter limit | Training stops before the first epoch if the model has more than 5,000,000 trainable parameters |

## Experiments and results

Each step was trained once. **Val.**: accuracy on the 5,000-image validation split with the final-epoch EMA weights
(raw weights for Step 1). **Flip**: the same with horizontal-flip averaging, the procedure used for the test
predictions from Step 2 on. **Public / Private**: Kaggle accuracy of the step's `submission.csv` on the two halves
(about 1,000 images each) of the hidden evaluation set. **—**: not applicable or not recorded.

| Step | Notebook | Change | Epochs | Train images | Val. (%) | Flip (%) | Public | Private |
|---|---|---|---|---|---|---|---|---|
| 1 | `step1_baseline.ipynb` | Baseline: crop and flip, cross-entropy, raw weights, no flip averaging | 200 | 45k | 95.40 | — | 0.849 | 0.821 |
| 2 | `step2_main.ipynb` | + TrivialAugmentWide, Cutout, label smoothing, EMA, flip averaging | 200 | 45k | 96.80 | 97.08 | 0.880 | 0.867 |
| 3b | `step3b_main_300ep_optional.ipynb` | Step 2 for 300 epochs (control for Step 3) | 300 | 45k | 97.06 | 97.18 | — | — |
| 3 | `step3_main_cutmix.ipynb` | Step 3b + CutMix | 300 | 45k | 97.32 | 97.58 | 0.891 | 0.867 |
| 4 | `step4_teacher.ipynb` | Teacher: WRN-28-10 with the Step 2 recipe | 200 | 45k | 97.62 | 97.92 | — | — |
| 5 | `step5_kd.ipynb` | Step 3b + distillation from the Step 4 teacher | 300 | 45k | 97.36 | 97.60 | 0.892 | 0.863 |
| 7 | `step7_teacher_cutmix.ipynb` | Teacher: WRN-28-10 with the Step 3 recipe | 300 | 45k | 97.82 | 97.94 | — | — |
| 8 | `step8_kd_ensemble_cutmix.ipynb` | CutMix + distillation; candidate teachers: Steps 4, 7 → average of both chosen | 600 | 45k | 97.76 | 97.88 | 0.901 | 0.877 |
| 9 | `step9_teacher_cutmix_600ep.ipynb` | Teacher: Step 7 for 600 epochs, training seed 1 | 600 | 45k | 97.92 | 98.24 | — | — |
| 10 | `step10_kd_3teachers.ipynb` | Step 8 recipe; candidate teachers: Steps 4, 7, 9 → Step 9 alone chosen | 1,000 | 45k | 98.04 | 98.14 | 0.907 | 0.875 |
| 11 | `step11_teacher_cutmix_600ep_seed2.ipynb` | Teacher: Step 9 with training seed 2 | 600 | 45k | —¹ | —¹ | — | — |
| 12 | `step12_kd_2teachers_tta.ipynb` | Step 10 recipe; candidate teachers: Steps 9, 11 → both chosen; each teacher's target averaged over the batch and its horizontal flip | 1,000 | 45k | 98.06 | 98.10 | 0.910 | 0.867 |
| 6a | `step6_final.ipynb` with `BEST = "kd_ens_cutmix"` | Step 8 configuration on all training images | 600 | 50k | — | — | 0.897 | 0.877 |
| 6b | `step6_final.ipynb` with `BEST = "kd3_cutmix"` | Step 10 configuration on all training images | 1,000 | 50k | — | — | 0.903 | 0.879 |
| **13** | `step13_kd_long_full.ipynb` | **Final model:** Step 10 configuration (Step 9 teacher) for 2,000 epochs on all training images | 2,000 | 50k | — | — | **0.918** | **0.881** |

¹ Not recorded: the Step 11 run folder was not copied into this repository.

**Teacher selection** (`select_teachers`, Steps 8, 10 and 12). Every candidate teacher and every average of two or more
candidates is evaluated on the validation split with flip averaging; the most accurate combination provides the
distillation targets (the smaller combination on a tie). The chosen teachers are saved in the run's `config.json`.

## Run records

`runs/` holds the record of every run except Step 11. The code writes each run to `runs/<name>/`, with `name` taken
from `CONFIG`; in this repository the record folders carry the step number as a prefix. Checkpoints (`last.pt`,
`best.pt`) are not committed (about 56 MB per student and 438 MB per teacher); the team keeps them and can provide them
on request.

| Step | Run name (`CONFIG["name"]`) | Record | GPU | Training time |
|---|---|---|---|---|
| 1 | `baseline` | [`runs/step1_baseline/`](runs/step1_baseline) | NVIDIA RTX 3060 Laptop | 1.34 h |
| 2 | `main` | [`runs/step2_main/`](runs/step2_main) | NVIDIA RTX 3060 Laptop | 1.27 h |
| 3 | `main_cutmix` | [`runs/step3_main_cutmix/`](runs/step3_main_cutmix) | NVIDIA A100-SXM4-40GB | 0.76 h |
| 3b | `main_300ep` | [`runs/step3b_main_300ep/`](runs/step3b_main_300ep) | NVIDIA A100-SXM4-40GB | 0.74 h |
| 4 | `teacher_wrn28_10` | [`runs/step4_teacher_wrn28_10/`](runs/step4_teacher_wrn28_10) | NVIDIA A100-SXM4-40GB | 0.70 h |
| 5 | `kd` | [`runs/step5_kd/`](runs/step5_kd) | NVIDIA A100-SXM4-40GB | 0.82 h |
| 7 | `teacher_cutmix` | [`runs/step7_teacher_cutmix/`](runs/step7_teacher_cutmix) | NVIDIA A100-SXM4-40GB | 1.04 h |
| 8 | `kd_ens_cutmix` | [`runs/step8_kd_ens_cutmix/`](runs/step8_kd_ens_cutmix) | NVIDIA A100-SXM4-40GB | 1.97 h |
| 9 | `teacher_cutmix_600ep` | [`runs/step9_teacher_cutmix_600ep/`](runs/step9_teacher_cutmix_600ep) | NVIDIA A100-SXM4-40GB | 2.08 h |
| 10 | `kd3_cutmix` | [`runs/step10_kd3_cutmix/`](runs/step10_kd3_cutmix) | NVIDIA A100-SXM4-40GB | 2.82 h |
| 11 | `teacher_cutmix_600ep_s2` | not included¹ | — | — |
| 12 | `kd_tta_cutmix` | [`runs/step12_kd_tta_cutmix/`](runs/step12_kd_tta_cutmix) | NVIDIA A100-SXM4-40GB | 4.91 h |
| 6a | `kd_ens_cutmix_full` | [`runs/step6_kd_ens_cutmix_full/`](runs/step6_kd_ens_cutmix_full) | NVIDIA A100-SXM4-40GB | 2.25 h |
| 6b | `kd3_cutmix_full` | [`runs/step6_kd3_cutmix_full/`](runs/step6_kd3_cutmix_full) | NVIDIA A100-SXM4-40GB | 3.11 h |
| 13 | `kd3_cutmix_long_full` | [`runs/step13_kd3_cutmix_long_full/`](runs/step13_kd3_cutmix_long_full) | NVIDIA A100-SXM4-40GB | 6.24 h |

Files in a record: `config.json` (settings, with the teachers after selection), `run_info.json` (GPU, PyTorch version,
parameter count), `log.csv` (per epoch: learning rate, training loss and accuracy, validation accuracy of the raw and
EMA weights, epoch time), `eval.json` (validation accuracy without and with flip averaging) and `submission.csv`.
Steps 1–2 were run with `py_scripts/` and have `config.yaml`, `log.csv` and `submission.csv`. Training time is the sum
of the `time_s` column of `log.csv` (training iterations only). The cluster runs used PyTorch 2.8.0+cu128 (Step 5:
2.6.0+cu124), as recorded in `run_info.json`.

## Reproducing the submitted predictions

### 1. Environment

```bash
pip install -r requirements.txt      # torch, torchvision, numpy, pillow, jupyter
```

Our cluster runs used PyTorch 2.8.0 (CUDA 12.8 build) on one NVIDIA A100-SXM4-40GB. A CUDA GPU is needed in practice
(see the training times above); mixed precision is used only on CUDA.

### 2. Data

Download the competition data (Kaggle → Data → Download All) and extract it into `data/raw/` at the repository root:

```
data/raw/
├── train_images/            50,000 PNG files
├── test_images/             2,000 PNG files
├── train_labels.csv         id, filename, label
├── test.csv                 id, filename
└── sample_submission.csv
```

The files may also sit one folder deeper. The first notebook run converts them into `data/cache/` (Section 3), which
all notebooks share. `data/` is excluded from git.

### 3. Running a step

Start Jupyter in the repository root (all paths are relative to it), open a step notebook and run all cells
(Kernel → Restart Kernel and Run All Cells), or run it without the interface:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 step9_teacher_cutmix_600ep.ipynb
```

A run writes `runs/<name>/`: `config.json`, `run_info.json`, `log.csv`, `last.pt`, `best.pt` (runs with a validation
split), `eval.json` and, for students, `submission.csv` (columns `id,label`, in the order of `sample_submission.csv`).
If a run is interrupted, running the notebook again continues from `last.pt`; a finished run is not trained again.
Distillation steps load the finished teachers from `runs/<teacher name>/last.pt` and stop with a message if a teacher
is missing or unfinished.

### 4. Final submission (Step 13)

```bash
# teacher: WRN-28-10, 600 epochs on the 45,000-image split (about 2.1 h on an A100)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 step9_teacher_cutmix_600ep.ipynb
# student: 2,000 epochs on all 50,000 images, distilled from the Step 9 teacher (about 6.2 h on an A100)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 step13_kd_long_full.ipynb
```

The predictions are written to `runs/kd3_cutmix_long_full/submission.csv`; the submitted file is
[`runs/step13_kd3_cutmix_long_full/submission.csv`](runs/step13_kd3_cutmix_long_full/submission.csv). Step 13 has no
validation split, so its `CONFIG` names the teacher directly: the Step 9 teacher, which Step 10 selected on the
validation split.

### 5. Other submitted predictions

| Submission | Train first | Then run | Output |
|---|---|---|---|
| Step 12 (second selected submission) | Steps 9, 11 | `step12_kd_2teachers_tta.ipynb` | `runs/kd_tta_cutmix/submission.csv` |
| Step 10 | Steps 4, 7, 9 | `step10_kd_3teachers.ipynb` | `runs/kd3_cutmix/submission.csv` |
| Step 8 | Steps 4, 7 | `step8_kd_ensemble_cutmix.ipynb` | `runs/kd_ens_cutmix/submission.csv` |
| Step 5 | Step 4 | `step5_kd.ipynb` | `runs/kd/submission.csv` |
| Step 3 | — | `step3_main_cutmix.ipynb` | `runs/main_cutmix/submission.csv` |
| Step 2 | — | `step2_main.ipynb` (or `py_scripts/`, see 6.) | `runs/main/submission.csv` |
| Step 1 | — | `step1_baseline.ipynb` (or `py_scripts/`, see 6.) | `runs/baseline/submission.csv` |
| Step 6a | Steps 4, 7 | `step6_final.ipynb` with `BEST = "kd_ens_cutmix"` | `runs/kd_ens_cutmix_full/submission.csv` |
| Step 6b | Steps 9, 10 | `step6_final.ipynb` with `BEST = "kd3_cutmix"` (the committed setting) | `runs/kd3_cutmix_full/submission.csv` |

Step 6 retrains the chosen configuration on all 50,000 images and reads the teachers selected by Step 8 or Step 10
from `runs/<BEST>/config.json`. To run Step 6b without re-running Step 10, copy `runs/step10_kd3_cutmix/config.json`
to `runs/kd3_cutmix/config.json`; only the Step 9 teacher is then needed. `BEST = "auto"` picks the configuration with
the highest flip-averaged validation accuracy found in `runs/<name>/eval.json`.

### 6. Steps 1–2 with the command-line scripts (as run)

The Step 1 and Step 2 records were produced with `py_scripts/`, which implements the same model, data pipeline and
settings as the Step 1–5 notebooks:

```bash
cd py_scripts
pip install -r requirements.txt                    # also installs pyyaml
python prepare_data.py --raw ../data/raw --out data/cache
python train.py --config configs/baseline.yaml     # Step 1 -> runs/baseline/
python predict.py --ckpt runs/baseline/last.pt --val --no-tta --out runs/baseline/submission.csv
python train.py --config configs/main.yaml         # Step 2 -> runs/main/
python predict.py --ckpt runs/main/last.pt --val --out runs/main/submission.csv
```

`run_step1_local.bat` and `run_step2_local.bat` run the same commands on Windows (edit the conda activation line for
your machine).

### 7. Predictions from a saved checkpoint

To regenerate a submission without training (for example from a checkpoint provided by the team), place the
checkpoint at `runs/<name>/last.pt`, run Sections 1–7 of the step's notebook, then:

```python
predict("runs/kd3_cutmix_long_full/last.pt", out="runs/kd3_cutmix_long_full/submission.csv")
```

`predict` loads the EMA weights, prints the parameter count, averages the predictions over horizontal flips and writes
the submission and `eval.json`.

### Determinism

Seeds are fixed, but GPU runs are not bit-exact (`cudnn.benchmark`, mixed precision, and a number of data-loading
workers that depends on the machine). A re-run therefore gives a model of the same accuracy up to run-to-run noise
rather than an identical file; the submitted files are kept in `runs/`.

## Compliance with the competition rules

- Only the 50,000 labeled training images provided for the competition are used for training; no external data.
- No pretrained weights: every network, including the WRN-28-10 teachers, is randomly initialized and trained from
  scratch on these images (the teachers on the 45,000-image split).
- The hidden evaluation (test) images are used only to write `submission.csv`; outside the Kaggle submission process
  they are not used for training, validation, normalization statistics or model selection.
- The final model is a single network with 4,696,394 trainable parameters; the teacher only provides training targets.
- No external code is copied into this repository. It uses PyTorch, torchvision (including
  `torchvision.transforms.TrivialAugmentWide`), NumPy and Pillow; the methods it implements are cited below.

## References

1. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016.
   https://arxiv.org/abs/1512.03385 (residual networks)
2. K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks," ECCV 2016.
   https://arxiv.org/abs/1603.05027 (pre-activation residual blocks)
3. S. Zagoruyko and N. Komodakis, "Wide Residual Networks," BMVC 2016. https://arxiv.org/abs/1605.07146
   (WRN-28-10 teacher)
4. T. He, Z. Zhang, H. Zhang, Z. Zhang, J. Xie, and M. Li, "Bag of Tricks for Image Classification with Convolutional
   Neural Networks," CVPR 2019. https://arxiv.org/abs/1812.01187 (ResNet-D shortcut, no weight decay on BatchNorm
   parameters and biases, learning-rate warm-up with cosine decay)
5. S. G. Müller and F. Hutter, "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation," ICCV 2021.
   https://arxiv.org/abs/2103.10158
6. T. DeVries and G. W. Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout,"
   arXiv:1708.04552, 2017. https://arxiv.org/abs/1708.04552
7. S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Regularization Strategy to Train Strong
   Classifiers with Localizable Features," ICCV 2019. https://arxiv.org/abs/1905.04899
8. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception Architecture for Computer
   Vision," CVPR 2016. https://arxiv.org/abs/1512.00567 (label smoothing)
9. G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.
   https://arxiv.org/abs/1503.02531
10. L. Beyer, X. Zhai, A. Royer, L. Markeeva, R. Anil, and A. Kolesnikov, "Knowledge Distillation: A Good Teacher Is
    Patient and Consistent," CVPR 2022. https://arxiv.org/abs/2106.05237 (teacher and student see the same augmented
    images; long distillation schedules)
11. PyTorch contributors, `torchvision.transforms.TrivialAugmentWide`, torchvision documentation.
    https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.TrivialAugmentWide.html
