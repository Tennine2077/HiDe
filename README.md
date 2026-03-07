# 🙈 HiDe

> **The official implementation of [HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling](https://arxiv.org/abs/2510.00054)**

<div align="center">

[![Paper](https://img.shields.io/badge/📜_Paper-ArXiv-red)](https://arxiv.org/abs/2510.00054)
[![License](https://img.shields.io/badge/📄_License-Apache_2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/🐍_Python-3.11+-blue)]()

</div>

---

## 📰 News

| Date | Update |
|:----:|:-------|
| 🔥 **2026/03/07** | Added support for Qwen3-VL! |
| 💻 **2025/12/15** | Code released on GitHub! Feel free to open issues if you have questions~ |
| 📕 **2025/10/28** | Paper released on ArXiv! |

---

## 🎯 What is HiDe?

Multimodal Large Language Models (MLLMs) have made significant strides in visual understanding tasks. However, their performance on **high-resolution images** remains suboptimal.

**Wait... is it really about "small objects"?** 🤔

Our analysis reveals a different story: the main issue is **not object size**, but rather **complex background interference**! 🎭

<div align="center">
<img src="https://arxiv.org/html/2510.00054/pics/method.png" width="90%" alt="HiDe Framework"/>
<p><em>Figure: Overview of the HiDe Framework (TAD + LPD)</em></p>
</div>

### 💡 Our Solution

We propose the **Hierarchical Decoupling Framework (HiDe)** — a **training-free** framework that includes:

| Component | What it does |
|:----------|:--------------|
| 🔍 **Token-wise Attention Decoupling (TAD)** | Decouples question tokens and identifies key information tokens, then leverages attention weights for precise alignment with target visual regions |
| ✂️ **Layout-Preserving Decoupling (LPD)** | Decouples target regions from background and reconstructs a compact representation while preserving essential spatial layouts |

---

## 🗂️ Repository Structure

```
HiDe/
├── Hide/
│   ├── Qwen2.5/              # 🤖 Qwen2.5-VL Implementation
│   │   ├── inference.py          # Core inference logic
│   │   ├── cycle_infer.py        # Multi-GPU inference entry
│   │   ├── Get_box.py            # Attention-based bounding box extraction
│   │   ├── Vstar_Metric.py       # Evaluation metrics
│   │   └── utiles.py             # Utility functions
│   │
│   ├── Qwen3/                # 🔥 Qwen3-VL Implementation
│   │   ├── inference.py
│   │   ├── cycle_infer.py
│   │   ├── Get_box.py
│   │   └── utiles.py
│   │
│   └── Internvl/             # 👁️ InternVL3 Implementation
│       ├── cycle_inference_internvl.py
│       └── utiles_internvl.py
│
├── requirements.txt          # 📦 Dependencies
├── LICENSE
└── README.md
```

---

## 🛠️ Installation

```bash
# Create conda environment
conda create -n HiDe python=3.11.4
conda activate HiDe

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

Prepare your dataset as a JSON file with the following structure:

```json
[
    {
        "id": "0",
        "question": "What is the material of the glove?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather",
        "labels": "A",
        "image_path": "vstar_bench/direct_attributes/sa_4690.jpg",
        "category": "direct_attributes"
    },
    {
        "id": "1",
        "question": "What is the color of the dustpan?\n(A) purple\n(B) red\n(C) blue\n(D) white",
        "labels": "C",
        "image_path": "vstar_bench/direct_attributes/sa_86101.jpg",
        "category": "direct_attributes"
    }
]
```

---

## 🚀 Quick Start

### For Qwen2.5-VL

```bash
cd Hide/Qwen2.5

# Run inference
python cycle_infer.py

# Calculate metrics
python Vstar_Metric.py
```

### For Qwen3-VL

```bash
cd Hide/Qwen3
python cycle_infer.py
```

### For InternVL3

```bash
cd Hide/Internvl
python cycle_inference_internvl.py
```

---

## ⚙️ Configuration

Key hyperparameters you can adjust in `cycle_infer.py`:

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `sigma` | Gaussian filter sigma for attention smoothing | `[3]` |
| `threshold` | Threshold for attention binarization | `[0.7]` |
| `max_pixels` | Maximum pixel count for image processing | `16384` |
| `Parallels` | Enable multi-GPU parallel processing | `True/False` |

---

## 📝 Citation

If you find HiDe helpful in your research, please consider citing:

```bibtex
@misc{liu2025hiderethinkingzoominmethod,
      title={HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling}, 
      author={Xianjie Liu and Yiman Hu and Yixiong Zou and Liang Wu and Jian Xu and Bo Zheng},
      year={2025},
      eprint={2510.00054},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.00054}, 
}
```

---

## 📄 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgements

Thanks to all the amazing open-source projects that made this possible:
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [InternVL](https://github.com/OpenGVLab/InternVL)

---

<div align="center">

**Made with ❤️ by the HiDe Team**

⭐ If you find this project useful, please give us a star! ⭐

</div>