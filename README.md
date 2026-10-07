# Sensor-Language-Action Models

[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2610.08244)
[![Webpage](https://img.shields.io/badge/website-project-blue)](https://yang-ai-lab.github.io/OpenSLA/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-OpenSLA-FFD21E)](https://huggingface.co/yang-ai-lab/OpenSLA)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](#-installation)


## 🔥 News

- **[2026-10-06]** Our paper is available on [arXiv](https://arxiv.org/abs/2610.08244).
- **[2026-10-06]** Code released on GitHub, and model released on [HuggingFace](https://huggingface.co/yang-ai-lab/OpenSLA).
- **[2026-10-06]** [Project website](https://yang-ai-lab.github.io/OpenSLA/) is live!

## 📖 Introduction

**OpenSLA** is a new family of models - _Sensor-Langauge-Action_ (SLA) models. It takes a multi-channel sensor history together with language context and
produces a structured action prediction and a sensor-state caption. We release
two variants:

- **OpenSLA-B**: sensor tokens are projected and prepended to the language
  prompt as a flat token prefix.
- **OpenSLA-H**: builds on the Base model with a hierarchical sensor encoder that compresses
  the signal into local, per-channel, and global memory tokens.


## 📖 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Pretrained Weights](#-pretrained-weights)
4. [Usage](#-usage)
5. [Datasets](#-datasets)
6. [Project Structure](#-project-structure)
7. [Citation](#-citation)

## 💿 Installation

```bash
git clone https://github.com/yang-ai-lab/OpenSLA.git
cd OpenSLA
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.5
- Transformers >= 5.8.1 (for the Qwen3.5 backbone)
- PEFT >= 0.12


## 🚀 Quick Start

`demo.ipynb` loads a checkpoint, builds an input batch, and runs prediction.
The same flow in Python:

```python
import torch
from opensla import OpenSLA, SensorBatch

model = OpenSLA.from_checkpoint(
    "pretrained_weights/opensla_h_clinical.pt", domain="clinical",
    dino_checkpoint="pretrained_weights/waveform_encoder.ckpt",
    numeric_config="pretrained_weights/numeric_config.json",
    action_group_types="pretrained_weights/action_group_types.json",
)

batch = torch.load("data/preprocessed_batch.pt", weights_only=True)
predictions = model.predict(batch["input_text"], SensorBatch(**batch["sensors"]))
```


## 📦 Pretrained Weights

| Model | Domain | Download |
|-------|--------|----------|
| OpenSLA-H | clinical | [OpenSLA](https://huggingface.co/yang-ai-lab/OpenSLA) |

The download includes the model checkpoint, the waveform-encoder checkpoint,
`numeric_config.json`, and `action_group_types.json`. Put them under
`pretrained_weights/`. 

## 👩‍💻 Usage

### Input Format

The model takes preprocessed signals as a `.pt` dictionary with `domain`,
`input_text` (one text context per sample), and `sensors`. The sensors are:

- **Waveform**: `waveform` of shape `[B, M, C, T]`, with `M` one-minute slots,
  `C` physical channels, and `T = 7500` samples per slot (60 s at 125 Hz), plus
  a boolean `waveform_channel_mask` (`[B, M, C]`) marking observed
  channel-minutes and int64 `waveform_modality_ids` (`[B, C]`) indexing
  `opensla.WAVEFORM_MODALITIES[domain]`.
- **Numeric**: observed events as parallel `[B, E]` tensors (`values`,
  `rel_time_min` in minutes before the decision, `measure_ids`, `source_ids`,
  `event_mask`), plus per-measure summary features (`summary_features`,
  `summary_measure_ids`).

### Command line

Point `configs/template.json` at your weights and input batch, then:

```bash
opensla --config configs/template.json --dry-run   # print the resolved config
opensla --config configs/template.json             # write outputs/predictions.jsonl
```

On Slurm:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION scripts/run_template.sbatch --config configs/template.json
```


## 📊 Datasets

The models are trained and evaluated on six datasets from three healthcare
settings, with an additional MIMIC-IV held out as an external clinical cohort. All of them are
publicly available and can be accessed through the following links.

| Dataset | Domain | Sensors | Source |
|---------|--------|---------|--------|
| MC-MED | Clinical | ECG, plethysmography, respiration, arterial pressure; vital signs, labs, ventilator and other charted measurements | [PhysioNet](https://physionet.org/content/mc-med/1.0.0/) |
| MIMIC-III | Clinical | same as MC-MED | [PhysioNet](https://physionet.org/content/mimiciii/1.4/) |
| MIMIC-IV | Clinical, external evaluation only | waveform-linked subset | [PhysioNet](https://physionet.org/content/mimiciv/2.0/), [waveforms](https://physionet.org/content/mimic4wdb/0.1.0/) |
| MOVER | Operating room | ECG, plethysmography, arterial/central venous pressure, capnography, airway pressure, EEG; vital signs, hemodynamics, ventilation gases, labs | [UCI MOVER](https://mover.ics.uci.edu/) |
| VitalDB | Operating room | same as MOVER | [vitaldb.net](https://vitaldb.net/dataset/), [PhysioNet](https://physionet.org/content/vitaldb/1.0.0/) |
| MetaboNet | CGM | glucose trace; basal insulin delivery | [metabo-net.org](https://metabo-net.org/) |
| PEDAP | CGM | glucose trace; basal insulin delivery | [Jaeb Center](https://public.jaeb.org/datasets/diabetes) |


## 📁 Project Structure

```
OpenSLA/
├── src/opensla/
│   ├── inference.py         # input format, checkpoint loading, sensor fusion, OpenSLA.predict
│   ├── model.py             # action heads, evidence selector, label ranker, caption decoder, multimodal LM
│   ├── sensors.py           # waveform/numeric encoders, hierarchical signal memory, token projectors
│   ├── vit1d.py             # waveform-encoder backbone (adapted from OSF)
│   └── cli.py               # `opensla` command
├── configs/template.json    # inference configuration
├── scripts/run_template.sbatch
├── pretrained_weights/      # downloaded checkpoints go here
├── docs/                    # project page (GitHub Pages)
└── demo.ipynb               # quick start demo
```


## 📝 Citation

If you use this code or models in your research, please cite our paper:

```bibtex
@misc{xu2026opensla,
  title  = {Sensor-Language-Action Models},
  author = {Xu, Yuekai and Shuai, Zitao and Yang, Yuzhe},
  journal = {arXiv preprint arXiv:2610.08244},
  year = {2026}
}
```

## Acknowledgments

The waveform encoder adapts from the sensor encoder from
[OSF](https://github.com/yang-ai-lab/OSF-Open-Sleep-FM) (MIT License).
