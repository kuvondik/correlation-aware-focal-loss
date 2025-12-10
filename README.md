# 📦 Dense Object Detection on SKU-110K

### RetinaNet Baseline + LocalRetinaNet (Correlation-Aware Focal Loss)

This repository provides a **complete training and evaluation pipeline** for dense object detection using:

### RetinaNet Baseline

- Built using `torchvision` (`retinanet_resnet50_fpn`)
- Anchor-based detection
- Standard focal loss

### LocalRetinaNet (Custom Model)

- ResNet-50 + FPN (P3–P5)
- Multi-level anchor generator
- Local correlation-aware focal loss (our proposed improvement)
- Supports training, inference, and COCO-style evaluation

---

# 🚀 Getting Started

This project includes two main files:

## 1. `gc_local_focal_loss.ipynb` (Google Colab Notebook)

- **Use this only on Google Colab**
- Dataset + pretrained models download automatically
- Just run the notebook from top to bottom
- Training cells are labeled `[Training]`

## 2. `local_focal_loss.py` (Local Environment Script)

- Use for macOS / Linux / Windows local execution
- Requires manual dataset extraction and dependency installation

---

# 📥 Dataset Setup (Local Environment)

1. **Download the modified SKU-110K dataset**
   https://drive.google.com/file/d/1QrZ6zTbOSiE28TQkBExb4Fa7EM6i5mfr/view?usp=drive_link
2. Extract into:
   
   correlation-aware-focal-loss/src/data/SKU110K_modified

---

# 📥 (Optional) Pretrained Models

3. Download pretrained weights:
   https://drive.google.com/file/d/1M8TJoZ-P8wiU1KLGlSDpRUggNavzCWia/view?usp=drive_link
   
   Extract inside:
   `correlation-aware-focal-loss/src/output/pt-models`

---

# 🛠 Install Dependencies

bash

```
pip install numpy torch torchvision pillow opencv-python matplotlib tqdm pycocotools
```

---

macOS ARM fallback:

```
pip install pycocotools-macos
```


---

# ▶️ Run Locally

`python local_focal_loss.py`

A menu allows training, inference, and evaluation.

---

# 📚 Citation

If you use this repository, please cite the following foundational works:

### Focal Loss for Dense Object Detection
Lin et al., ICCV 2017

```
@inproceedings{lin2017focal,
    title={Focal Loss for Dense Object Detection},
    author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    pages={2980--2988},
    year={2017}
}
```
### The SKU-110K Dataset
Goldman et al., CVPR 2019

```
@inproceedings{sku110k,
  title={Precise Detection in Densely Packed Scenes},
  author={Goldman, Erez and Herzig, Roei and Eisenschtat, Adi and Goldberger, Jacob and Konen, Eli},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9639--9648},
  year={2019}
}
```
