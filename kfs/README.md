## Preparation
We train Key Frame Selector based on GLUS. Please follow [GLUS_README](../README.md) to prepare data and train the GLUS model.

### Data annotation
Please follow `Evaluation` section of [GLUS_README](../README.md) to get inference masks and predicted ious on `Mevis_train` and `Youtube_train`. An example

```
scripts/inference.sh
python utils/eval_mevis.py \
  --mevis_exp_path='$GLUS_ROOT/data/mevis/train/meta_expressions.json' \
  --mevis_mask_path='$GLUS_ROOT/data/mevis/train/mask_dict.json'
  --mevis_pred_path='$GLUS_ROOT/generated'
  --iou_path='$GLUS_ROOT/data/train_iou/mevis'
```
Run this in and predicted ious of train dataset will be saved to `iou_path`.

Use `python kfs/utils/merge_folder.py` to copy Mevis annotation and RefYoutube annotation into one folder.

Set the `RVOS_ROOT` in `kfs/utils/dataset_config.py` before next steps.

### Model Weights
We use ChatUniVi as an init model. Follow the guidance to prepare for the pretrained weights of ChatUniVi:

1. Download the pretrained weights of Chat-UniVi from [Chat-UniVi-7b](https://huggingface.co/Chat-UniVi/Chat-UniVi).
  

<summary> Then organize them in the following architecture: </summary>

```
WEIGHTS_ROOT
├── Chat-UniVi
```
We recommend to set ``WEIGHTS_ROOT`` to ``GLUS_ROOT/checkpoints``.

## Training
Set the paths in the scripts and then run ``kfs/scripts/train_kfs.sh``. The scripts will automatically start the training, and transform the saved checkpoint into hugging-face format when the training finished.


## Inference
Set the paths in ``kfs/scripts/inference_kfs.sh``, and then run it. It will predict score for each frame in validation set  and then choose the most confident (important) frame and save it in json files.

For final inference of RVOS task, please follow `evaluation with key frame` in [GLUS_README](../README.md).

**Note:** 
* We use intermediate saved checkpoint for data annotation (about 15-20 Epoch training of GLUS). 
* We only test KFS on Mevis and Ref-Youtube-VOS from **GLUS<sup><i>S</i></sup>** setting, feel free to test the pipeline on more dataset and models.


## Todo list
  
- [x] Release KFS pipeline

- [x] Release KFS json for inference (GLUS_S)

- [ ] Release KFS model checkpoint


## Acknowledgement
This pipeline is built upon [Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi) and [VISA](https://github.com/cilinyan/VISA).