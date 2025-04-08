# GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation
<div style="text-align: center;">
  <p>
    <a href="https://openreview.net/profile?id=~Lang_Lin3">Lang Lin</a>*, 
    <a href="https://openreview.net/profile?id=~Xueyang_Yu1">Xueyang Yu</a>*, 
    <a href="https://ziqipang.github.io/">Ziqi Pang</a>*, 
    <a href="https://yxw.web.illinois.edu/">Yu-Xiong Wang</a>
  </p>
</div>

[[`Project Page`](https://glus-video.github.io/)] [[`arXiv(Coming soon)`]()]


[![Project](https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green)](https://glus-video.github.io/) 


<div align=center>
<img src="assets/teaserfig.png" style="width:100%;">
</div>

## Overview

**RefVOS in complex scenarios** places high demands on models' video understanding and fine-grained localization capabilities. Recently, numerous models leveraging **MLLM-based** comprehension and reasoning abilities have been proposed to address this challenge. Our **GLUS** advances further along this methodological path.

🚀 **GLUS is principled.** It utilizes global-local reasoning to combine holistic video understanding with detailed frames understanding, unleashing the potential of fine-grained segmentation in complex scenarios.

✨ **GLUS is powerful.** It unifies the methods of memory bank, object contrastive learning and key frame selection to tackle the problems of mask inconsistency and object obfuscation, achieving state-of-the-art performance in complex-scenario RefVOS tasks.

📌 **GLUS is simple.** It elegantly integrates the approach for complex-scenario RefVOS tasks within a single MLLM framework, eliminating the necessity of utilizing other independent modules.

<div align=center>
<img src="assets/pipeline_00.png" style="width:100%;">
</div>

## News

## Installation
```shell
git clone git@github.com:GLUS-video/GLUS.git && cd GLUS
pip install -r requirements.txt
pip install ./model/segment-anything-2
pip install flash-attn==2.6.2 --no-build-isolation
```

## Model Zoo

For more convenient following, we provide the checkpoints of GLUS without object contrastive learning.

| Model                           | Training Datasets          | Methods             | Download |  MeViS J\&F | Ref-Youtube-VOS J\&F |
|--------------------------------------|---------------------------------|--------------|----------|-----------|-----------------------|
| **GLUS<sup><i>S</i></sup><sub>partial</sub>** | MeViS, Ref-Youtube-VOS          | GLU + MB |  [HuggingFace](https://huggingface.co/Swindl/GLUS-S-partial/tree/main), [ModelScope](https://www.modelscope.cn/models/LangLin/GLUS-S-partial/files)        | 49.5 | 65.2 |
| **GLUS<sup><i>S</i></sup>**            | MeViS, Ref-Youtube-VOS          | GLU + MB + OC + KFS |  [HuggingFace](https://huggingface.co/Swindl/GLUS-S/tree/main), [ModelScope](https://www.modelscope.cn/models/LangLin/GLUS-S/files)        | 50.3 | 66.6 |
| **GLUS<sup><i>A</i></sup>**            | + RefDAVIS17, ReVOS, LVVIS      | GLU + MB |  [HuggingFace](https://huggingface.co/Swindl/GLUS-A/tree/main), [ModelScope](https://www.modelscope.cn/models/LangLin/GLUS-A/files)        | 51.3 | 67.3 |

Notes: “GLU”: Global-local unification, “MB”: End-to-end memory bank, “OC”: Object contrastive loss, “KFS”: key frame selection.

We recommend to download and store the pretrained weights at ``GLUS_ROOT/checkpoints``.

## Training and Validation

### 1. Data Preparation

Please follow the below architecture to prepare the datasets. We recommend to set ``DATASET_ROOT``  to ``GLUS_ROOT/data``.

1. RefVOS Datasets: [MeViS](https://github.com/henghuiding/MeViS), [Refer-YouTube-VOS](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-get-data), [Ref-DAVIS17](https://github.com/wjn922/ReferFormer/blob/main/docs/data.md).
2. Reasoning VOS Datasets: [ReVOS](https://github.com/cilinyan/ReVOS-api), [ReasonVOS](https://github.com/showlab/VideoLISA/blob/main/BENCHMARK.md)
3. Open-Vocabulary Video Instance Segmentation Dataset: [LV-VIS](https://github.com/haochenheheda/LVVIS/tree/main).

<details open>
<summary> <strong>Datasets Architecture</strong> </summary>

```
DATASET_ROOT
├── mevis
│   ├── train
│   │   ├── JPEGImages
│   │   ├── mask_dict.json
│   │   └── meta_expressions.json
│   ├── valid
│   │   ├── JPEGImages
│   │   └── meta_expressions.json
│   └── valid_u
│       ├── JPEGImages
│       ├── mask_dict.json
│       └── meta_expressions.json
├── Refer-YouTube-VOS
│   ├── meta_expressions
│   │   ├── train/meta_expressions.json
│   │   └── valid/meta_expressions.json
│   ├── train
│   │   ├── JPEGImages
│   │   └── Annotations
│   └── valid
│       └── JPEGImages
├── DAVIS17
│   ├── meta_expressions
│   │   ├── train/meta_expressions.json
│   │   └── valid/meta_expressions.json
│   ├── train
│   │   ├── JPEGImages
│   │   └── Annotations
│   └── valid
│       ├── JPEGImages
│       └── Annotations
├── LVVIS
│   ├── train
│   │   └── JPEGImages
│   ├── mask_dict.json
│   └── meta_expressions.json
├── ReVOS
│   ├── JPEGImages 
│   ├── mask_dict.json             
│   ├── mask_dict_foreground.json   
│   ├── meta_expressions_train_.json 
│   └── meta_expressions_valid_.json 
├── ReasonVOS
│   ├── JPEGImages 
│   ├── Annotations           
│   ├── meta_expressions.json 

```

</details>

### 2. Model Weights Preparation

Follow the guidance to prepare for the pretrained weights of LISA and SAM-2 for training GLUS:

1. Download the pretrained weights of LISA from [LISA-7B-v1](https://huggingface.co/xinlai/LISA-7B-v1/tree/main).
2. Download the pretrained weights of SAM-2 from [sam2_hiera_large](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt).
  
<details>
<summary> Then organize them in the following architecture: </summary>

```
WEIGHTS_ROOT
├── LISA-7B-v1
└── sam2_hiera_large.pt
```
   
We recommend to set ``WEIGHTS_ROOT`` to ``GLUS_ROOT/checkpoints``.

</details>

### 3. Training

Set the paths in the scripts and then run ``scripts/train_glus_s.sh`` or ``scripts/train_glus_a.sh``. The scripts will automatically start the training, and transform the saved checkpoint into hugging-face format when the training finished.

#### Key Frame Selection
For the usage of key frame selection, please refer to the [KFS_README](kfs/README.md).


### 4. Evaluation

Set the paths, ``val_set`` and ``set_name`` in ``scripts/inference.sh``, and then run it. It will detect the available GPUs firstly and then individually run parallelizable inference on each gpu.

#### Evaluation with Key Frame Selection
Set the args ``use_kf`` and ``kf_path`` in ``scripts/inference_kf.sh``, and then run it. We provide our json file on Mevis and Refyoutube-VOS for **GLUS<sup><i>S</i></sup>** on the [google drive](https://drive.google.com/drive/folders/1NcjOguZUmal7Xk7rihyhvs5GRK_RzQSO?usp=sharing).

After the masks are generated completely, run the corresponding evalaution python file in ``utils``. You may need to set the groundtruth mask path, predicted mask path and expressions json file path. Please refer to the eval files to see the help on arguments.

An example:

```
python utils/eval_mevis.py \
  --mevis_exp_path='$GLUS_ROOT/data/mevis/valid_u/meta_expressions.json' \
  --mevis_mask_path='$GLUS_ROOT/data/mevis/valid_u/mask_dict.json'
  --mevis_pred_path='$GLUS_ROOT/generated'
```

Specially, to evaluate the performance on ``Refer-YouTube-VOS Valid`` or ``MeViS Valid`` benchmarks, you may need to submit the predicted masks results following the guidance at [MeViS-Evaluation-Server](https://codalab.lisn.upsaclay.fr/competitions/15094) or [RefYoutube-Evaluation-Server](https://codalab.lisn.upsaclay.fr/competitions/3282).

## Inference and Demo

Please refer to ``demo.ipynb`` to inference on your own videos and referrings. An example is shown below.

Referring: The panda that has stayed in place with little movement.

For more examples, please refer to our [Project Page](https://glus-video.github.io/).

## Acknowledgement
We thank the contributors to the following open-source projects.  Our project is impossible without the inspirations from these excellent researchers.
* [LISA](https://github.com/dvlab-research/LISA)
* [SAM2](https://github.com/facebookresearch/sam2)
* [Mevis](https://github.com/henghuiding/MeViS)
* [VISA](https://github.com/cilinyan/VISA)
