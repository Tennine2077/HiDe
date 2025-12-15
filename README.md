# HiDe

The official implementation of [HiDe](https://arxiv.org/abs/2510.00054).

> # HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling
> 
> Xianjie Liu, Yiman Hu, Yixiong Zou, Liang Wu, Jian Xu, Bo Zheng
>
>💻2025/12/15: We released the code on GitHub, if you have any questions, ask us by issues~ 
>
> 📕2025/10/28: We released the paper on the ArXiv.
> 
> # Abstract
> Multimodal Large Language Models (MLLMs) have made significant strides in visual understanding tasks. However, their performance on high-resolution images remains suboptimal. While existing approaches often attribute this limitation to perceptual constraints and argue that MLLMs struggle to recognize small objects, leading them to use "zoom in" strategies for better detail, our analysis reveals a different cause: the main issue is not object size, but rather caused by complex background interference. We systematically analyze this "zoom in" operation through a series of decoupling experiments and propose the Hierarchical Decoupling Framework (HiDe), a training-free framework that uses Token-wise Attention Decoupling (TAD) to decouple the question tokens and identify the key information tokens, then leverages their attention weights to achieve precise alignment with the target visual regions. Subsequently, it employs Layout-Preserving Decoupling (LPD) to decouple these regions from the background and reconstructs a compact representation that preserves essential spatial layouts while eliminating background interference. HiDe sets a new SOTA on V\*Bench, HRBench4K, and HRBench8K, boosting Qwen2.5-VL 7B and InternVL3 8B to SOTA (92.1% and 91.6% on V\*Bench), even surpassing RL methods. After optimization, HiDe uses 75% less memory than the previous training-free approach.

<img width="1263" height="609" alt="image" src="https://github.com/user-attachments/assets/64b7b8bd-ab35-4f88-af75-2361d1141e1d" />

## Installation
```
conda create -n HiDe python = 3.11.4
conda activate HiDe

pip install -r requirements.txt
```

## Dataset Preparation
The structure of the "**data**" should be json as follows:
```
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
    },
]
```

# Test and metric
Run
```
cd HiDe
python cycle_infer.py
python Vstar_Metric.py
```

Please consider to cite HiDe if it helps your research.
```
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
