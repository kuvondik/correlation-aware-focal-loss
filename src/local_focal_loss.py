import os
import json
import math
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.transforms import functional as F
from torchvision.ops import box_iou

from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ======================================================================================
# UTILS
# ======================================================================================

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def fix_coco_json(path):
    """Add missing fields so pycocotools does not crash on annotation JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    # Prediction JSON is a list -> ignore
    if isinstance(data, list):
        return

    changed = False

    if "info" not in data:
        data["info"] = {}
        changed = True

    if "licenses" not in data:
        data["licenses"] = []
        changed = True

    if changed:
        with open(path, "w") as f:
            json.dump(data, f)


def pad_images_to_batch(images):
    """
    Pad list of CHW images to max height and max width within the batch.
    Keeps content in top-left, pads bottom/right with zeros.
    """
    import torch.nn.functional as F_pad

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = []
    for img in images:
        _, h, w = img.shape
        pad_bottom = max_h - h
        pad_right = max_w - w
        img_padded = F_pad.pad(img, (0, pad_right, 0, pad_bottom))
        padded.append(img_padded)

    return torch.stack(padded)


# ======================================================================================
# CONFIGURATION
# ======================================================================================

class Config:
    def __init__(self):
        # Dataset paths
        self.train_images = "./data/SKU110K_modified/images"
        self.val_images = "./data/SKU110K_modified/images"
        self.test_images = "./data/SKU110K_modified/images"

        self.train_annotations = "./data/SKU110K_modified/annotations/COCO_json/annotations_train.json"
        self.val_annotations = "./data/SKU110K_modified/annotations/COCO_json/annotations_val.json"
        self.test_annotations = "./data/SKU110K_modified/annotations/COCO_json/annotations_test.json"

        # Inference
        self.infer_image_path = "./data/SKU110K_modified/images/test_0.jpg"

        # Local RetinaNet (correlation-aware) hyperparameters
        self.num_classes_local = 1        # single "product" class
        self.lambda_reg_local = 0.1       # weight between cls and reg for local model
        self.num_epochs_local = 5
        self.batch_size_local = 1         # memory-safe for LocalRetinaNet

        # Baseline RetinaNet hyperparameters
        self.num_classes_retina = 2       # torchvision RetinaNet classes (background + product)
        self.num_epochs_retina = 5
        self.batch_size_retina = 2

        self.lr = 1e-4
        self.num_workers = 4

        # Save paths
        self.save_local_model_path = "./output/pt-models/retinanet_local_sku110k.pth"
        self.save_retinanet_model_path = "./output/pt-models/retinanet_sku110k.pth"

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )


config = Config()


# ======================================================================================
# DATASET (COCO FORMAT)
# ======================================================================================

class SKU110K_COCO(Dataset):
    def __init__(self, root, annotation_json, transforms=None):
        self.root = root
        self.transforms = transforms

        with open(annotation_json, "r") as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.ids = sorted(self.images.keys())

        self.annos = {img_id: [] for img_id in self.ids}
        for ann in data["annotations"]:
            self.annos[ann["image_id"]].append(ann)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.images[img_id]

        fname = info["file_name"].split("/")[-1]
        img_path = os.path.join(self.root, fname)
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for ann in self.annos[img_id]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # single product class (id=1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(int(img_id)),
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


# ======================================================================================
# SIMPLE TRANSFORMS FOR LOCAL RETINANET (NO RESIZE – KEEP CLEAN)
# ======================================================================================

class ToTensorOnly:
    """Transform that converts PIL to tensor and leaves boxes unchanged."""
    def __call__(self, img, target):
        img = F.to_tensor(img)
        return img, target


# ======================================================================================
# BASELINE RETINANET (torchvision)
# ======================================================================================

def create_retinanet(num_classes):
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors

    # replace cls head to match num_classes
    model.head.classification_head.num_classes = num_classes
    model.head.classification_head.cls_logits = nn.Conv2d(
        256, num_anchors * num_classes, kernel_size=3, padding=1
    )

    torch.nn.init.normal_(model.head.classification_head.cls_logits.weight, std=0.01)
    torch.nn.init.constant_(model.head.classification_head.cls_logits.bias, -4.0)
    return model


def train_one_epoch_retinanet(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
            }
            for t in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(
                f"[RetinaNet-Baseline][Epoch {epoch}] Step {step} "
                f"loss: {loss.item():.4f}"
            )

    print(f"{timestamp()} — RetinaNet Baseline Epoch {epoch} Avg Loss: {total_loss / len(loader):.4f}")


def run_inference_retinanet(model_path, image_path, config, save_output=False):
    device = config.device
    print(f"Loading RetinaNet baseline: {model_path}")

    model = create_retinanet(config.num_classes_retina)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Image not found: " + image_path)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).to(device)

    with torch.no_grad():
        out = model([tensor])[0]

    vis = img_bgr.copy()
    for box, score in zip(out["boxes"], out["scores"]):
        if score < 0.4:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if save_output:
        ensure_dir("./output/inference_retina_baseline")
        out_path = "./output/inference_retina_baseline/" + Path(image_path).stem + "_retina_pred.jpg"
        cv2.imwrite(out_path, vis)
        print("Saved:", out_path)

    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def batch_inference_retinanet(model_path, folder, config,
                              save_dir="./output/inference_retina_baseline"):
    device = config.device
    print("\n=== Batch Inference: RetinaNet Baseline ===")
    ensure_dir(save_dir)

    model = create_retinanet(config.num_classes_retina)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    with torch.no_grad():
        for file in tqdm(image_files, desc="RetinaNet Baseline Batch Infer"):
            path_img = os.path.join(folder, file)
            img_bgr = cv2.imread(path_img)
            if img_bgr is None:
                print("Skipping unreadable image:", file)
                continue

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = F.to_tensor(rgb).to(device)

            out = model([tensor])[0]
            vis = img_bgr.copy()

            for box, score in zip(out["boxes"], out["scores"]):
                if score < 0.4:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            save_path = os.path.join(save_dir, f"{Path(file).stem}_retina_pred.jpg")
            cv2.imwrite(save_path, vis)

    print("\n=== RetinaNet Baseline Batch Inference Completed ===")


# ======================================================================================
# LOCAL CORRELATION-AWARE FOCAL LOSS (for RetinaNet classification)
# ======================================================================================

class LocalFocalLoss2d(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, kernel_size=5, lambda_local=1.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.kernel_size = kernel_size
        self.lambda_local = lambda_local
        self.reduction = reduction
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, C, H, W), 0/1
        """
        logits = logits.float()
        targets = targets.float()

        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        eps = 1e-8
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        base_loss = -focal_weight * torch.log(p_t.clamp(min=eps))

        # spatial hardness
        base_loss_spatial = base_loss.mean(dim=1, keepdim=True)
        local_hardness = self.avg_pool(base_loss_spatial)

        global_mean = local_hardness.mean().detach()
        h_norm = local_hardness / (global_mean + eps)

        local_weight = 1 + self.lambda_local * (h_norm - 1)
        local_weight = torch.clamp(local_weight, min=0.1, max=3.0)
        local_weight = local_weight.expand_as(base_loss)

        final_loss = local_weight * base_loss

        if self.reduction == "mean":
            return final_loss.mean()
        elif self.reduction == "sum":
            return final_loss.sum()
        return final_loss


# ======================================================================================
# BOX ENCODING / NMS / ANCHORS
# ======================================================================================

def encode_boxes(anchors, gt_boxes):
    """
    anchors: (N,4), gt_boxes: (N,4)
    format: x1,y1,x2,y2
    """
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    aw = (anchors[:, 2] - anchors[:, 0])
    ah = (anchors[:, 3] - anchors[:, 1])

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gw = (gt_boxes[:, 2] - gt_boxes[:, 0])
    gh = (gt_boxes[:, 3] - gt_boxes[:, 1])

    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(anchors, deltas):
    """
    anchors: (N,4), deltas: (N,4) tx,ty,tw,th
    """
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    aw = (anchors[:, 2] - anchors[:, 0])
    ah = (anchors[:, 3] - anchors[:, 1])

    tx, ty, tw, th = deltas.unbind(dim=1)

    gx = tx * aw + ax
    gy = ty * ah + ay
    gw = aw * torch.exp(tw)
    gh = ah * torch.exp(th)

    x1 = gx - gw / 2
    y1 = gy - gh / 2
    x2 = gx + gw / 2
    y2 = gy + gh / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def assign_anchors_to_gt(anchors, gt_boxes, iou_pos_thresh=0.5, iou_neg_thresh=0.4):
    A = anchors.size(0)
    device = anchors.device

    labels = torch.full((A,), -1, dtype=torch.int64, device=device)
    matched_gt_boxes = torch.zeros((A, 4), dtype=torch.float32, device=device)

    if gt_boxes.numel() == 0:
        labels[:] = 0
        return labels, matched_gt_boxes

    ious = box_iou(anchors, gt_boxes)
    max_iou, max_idx = ious.max(dim=1)

    labels[max_iou < iou_neg_thresh] = 0
    labels[max_iou >= iou_pos_thresh] = 1

    matched_gt_boxes[:] = gt_boxes[max_idx]
    return labels, matched_gt_boxes


def nms(boxes, scores, threshold=0.5):
    return torchvision.ops.nms(boxes, scores, threshold)


class MultiLevelAnchorGenerator:
    """
    Multi-scale anchors for FPN levels P3..P5.
    sizes_per_level: list of list, e.g.
      [
        [32, 45, 64],
        [64, 90, 128],
        [128, 181, 256],
      ]
    ratios: list, e.g. [0.5, 1.0, 2.0]
    strides: list, e.g. [8, 16, 32]
    """
    def __init__(self, sizes_per_level, ratios, strides):
        self.sizes_per_level = sizes_per_level
        self.ratios = ratios
        self.strides = strides

    def _grid_anchors(self, grid_size, stride, sizes, device):
        H, W = grid_size
        shifts_x = (torch.arange(W, device=device) + 0.5) * stride
        shifts_y = (torch.arange(H, device=device) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        anchor_boxes = []
        for size in sizes:
            for ratio in self.ratios:
                size = float(size)
                ratio = float(ratio)
                w = size * math.sqrt(ratio)
                h = size / math.sqrt(ratio)
                anchors = torch.stack([
                    shift_x - w / 2,
                    shift_y - h / 2,
                    shift_x + w / 2,
                    shift_y + h / 2
                ], dim=1)
                anchor_boxes.append(anchors)

        return torch.cat(anchor_boxes, dim=0)

    def __call__(self, feature_shapes, device):
        """
        feature_shapes: list of (H, W) per level
        returns: list of Tensors (N_level, 4)
        """
        anchors_per_level = []
        for (H, W), stride, sizes in zip(feature_shapes, self.strides, self.sizes_per_level):
            anchors = self._grid_anchors((H, W), stride, sizes, device)
            anchors_per_level.append(anchors)
        return anchors_per_level


# ======================================================================================
# BACKBONE + FPN FOR LOCAL RETINANET (P3–P5 ONLY)
# ======================================================================================

class ResNetFPN(nn.Module):
    """
    ResNet50 backbone with FPN producing P3, P4, P5.
    Strides: P3=8, P4=16, P5=32.
    """
    def __init__(self, backbone_name="resnet50"):
        super().__init__()
        if backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(weights="DEFAULT")
            c3_channels = 512
            c4_channels = 1024
            c5_channels = 2048
        else:
            backbone = torchvision.models.resnet18(weights=None)
            c3_channels = 128
            c4_channels = 256
            c5_channels = 512

        # ResNet stem + stages
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # C2
        self.layer2 = backbone.layer2  # C3
        self.layer3 = backbone.layer3  # C4
        self.layer4 = backbone.layer4  # C5

        # Lateral 1x1 convs to 256 channels
        self.lateral3 = nn.Conv2d(c3_channels, 256, 1)      # C3 -> P3
        self.lateral4 = nn.Conv2d(c4_channels, 256, 1)      # C4 -> P4
        self.lateral5 = nn.Conv2d(c5_channels, 256, 1)      # C5 -> P5

        # 3x3 FPN convs
        self.p3_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.p4_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.p5_conv = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, x):
        # Bottom-up
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)   # stride 4
        c3 = self.layer2(c2)   # stride 8
        c4 = self.layer3(c3)   # stride 16
        c5 = self.layer4(c4)   # stride 32

        # Top-down FPN – use size=... to avoid any odd/odd mismatch
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + Fnn.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lateral3(c3) + Fnn.interpolate(p4, size=c3.shape[-2:], mode="nearest")

        # Smooth
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        # Only P3–P5
        return [p3, p4, p5]


# ======================================================================================
# RETINANET HEAD
# ======================================================================================

class RetinaNetHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # classification subnet
        cls_layers = []
        for _ in range(4):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            cls_layers.append(nn.ReLU(inplace=True))
        self.cls_subnet = nn.Sequential(*cls_layers)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)

        # bbox subnet
        box_layers = []
        for _ in range(4):
            box_layers.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            box_layers.append(nn.ReLU(inplace=True))
        self.box_subnet = nn.Sequential(*box_layers)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        # init
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -4.0)

        torch.nn.init.normal_(self.bbox_pred.weight, std=0.01)
        torch.nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, features):
        cls_logits = []
        bbox_regs = []
        for f in features:
            cls = self.cls_subnet(f)
            cls_logits.append(self.cls_logits(cls))

            box = self.box_subnet(f)
            bbox_regs.append(self.bbox_pred(box))
        return cls_logits, bbox_regs


# ======================================================================================
# LOCAL RETINANET (P3–P5, MULTI-LEVEL, BATCHED TRAINING)
# ======================================================================================

class LocalRetinaNet(nn.Module):
    """
    RetinaNet-like detector with:
    - ResNet50-FPN backbone (P3–P5 only, strides 8/16/32)
    - Multi-level anchors
    - LocalFocalLoss2d for classification
    - SmoothL1 for regression
    - torchvision-style API: model(list[Tensor]) -> list[dict]
    """
    def __init__(self, num_classes=1, lambda_reg=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg

        # Backbone with P3–P5
        self.backbone = ResNetFPN("resnet50")

        # Anchors: 3 scales per level, 3 ratios => 9 anchors per location
        self.num_anchors = 9
        self.head = RetinaNetHead(256, self.num_anchors, num_classes)

        # Anchor sizes and strides for P3, P4, P5
        base_sizes = [32, 64, 128]   # P3–P5
        scales = [1.0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        sizes_per_level = [[b * s for s in scales] for b in base_sizes]
        ratios = [0.5, 1.0, 2.0]
        strides = [8, 16, 32]

        self.anchor_gen = MultiLevelAnchorGenerator(sizes_per_level, ratios, strides)
        self.cls_loss_fn = LocalFocalLoss2d(alpha=0.25, gamma=2.0, lambda_local=1.0)
        self.reg_loss_fn = nn.SmoothL1Loss(reduction="sum")

    # ------------------------------------------------------------------
    # SINGLE-IMAGE LOSS (used inside batched forward)
    # ------------------------------------------------------------------
    def _compute_loss_single_image(
        self,
        cls_logits_list,
        bbox_regs_list,
        anchors_per_level,
        gt_boxes,
        feature_shapes,
        b,
        device,
    ):
        """
        cls_logits_list: list of [B, A*C, H, W] for levels P3..P5
        bbox_regs_list:  list of [B, A*4, H, W]
        anchors_per_level: list of (N_l, 4)
        gt_boxes: (M, 4)
        b: image index in batch
        """
        num_levels = len(cls_logits_list)
        level_offsets = []
        logits_flat_by_level = []
        box_flat_by_level = []

        start = 0
        for lvl in range(num_levels):
            cls_l_b = cls_logits_list[lvl][b]   # (A*C, H, W)
            box_l_b = bbox_regs_list[lvl][b]    # (A*4, H, W)
            _, Hf, Wf = cls_l_b.shape
            A = self.num_anchors

            N_l = Hf * Wf * A

            # flatten to (N_l, C) and (N_l, 4)
            cls_flat = cls_l_b.permute(1, 2, 0).reshape(N_l, self.num_classes)
            box_flat = box_l_b.permute(1, 2, 0).reshape(N_l, 4)

            logits_flat_by_level.append(cls_flat)
            box_flat_by_level.append(box_flat)

            end = start + N_l
            level_offsets.append((start, end, Hf, Wf))
            start = end

        # concat over levels
        logits_all = torch.cat(logits_flat_by_level, dim=0)  # (N_total, C)
        box_all = torch.cat(box_flat_by_level, dim=0)        # (N_total, 4)
        anchors_all = torch.cat([a.to(device) for a in anchors_per_level], dim=0)

        # assign anchors
        labels, matched_gt_boxes = assign_anchors_to_gt(anchors_all, gt_boxes)
        fg_mask = labels == 1

        # make classification targets (binary class: 0=bg, 1=product)
        cls_target_all = torch.zeros_like(logits_all)  # (N_total, C)
        if self.num_classes == 1:
            cls_target_all[fg_mask, 0] = 1.0
        else:
            raise NotImplementedError("Current pipeline assumes num_classes=1.")

        # ---- Local Focal Loss per-level (correlation-aware) ----
        cls_loss = torch.tensor(0.0, device=device)
        for lvl in range(num_levels):
            start, end, Hf, Wf = level_offsets[lvl]
            targets_lvl_1d = cls_target_all[start:end]  # (N_l, C)

            # reshape to (1, A*C, H, W) to match cls_logits
            t_map = targets_lvl_1d.reshape(
                Hf, Wf, self.num_anchors * self.num_classes
            ).permute(2, 0, 1).unsqueeze(0)

            logits_lvl = cls_logits_list[lvl][b].unsqueeze(0)  # (1, A*C, H, W)
            cls_loss = cls_loss + self.cls_loss_fn(logits_lvl, t_map)

        # ---- Regression loss on positives ----
        num_pos = int(fg_mask.sum().item())
        if num_pos > 0:
            pred_pos = box_all[fg_mask]  # (N_pos, 4)
            tgt_pos = encode_boxes(anchors_all[fg_mask], matched_gt_boxes[fg_mask])
            reg_loss_sum = self.reg_loss_fn(pred_pos, tgt_pos)  # summed over positives
        else:
            reg_loss_sum = torch.tensor(0.0, device=device)

        return cls_loss, reg_loss_sum, num_pos

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------
    def forward(self, images, targets=None):
        """
        images: list of Tensor (3, H, W)
        targets: list of dicts with key "boxes" (only used during training)
        """
        device = images[0].device

        # INFERENCE MODE (no targets)
        if not self.training or targets is None:
            return self._forward_inference(images)

        # TRAINING MODE
        x = pad_images_to_batch(images).to(device)  # (B, 3, Hmax, Wmax)
        B = x.shape[0]

        features = self.backbone(x)  # [P3, P4, P5], each (B, 256, H, W)
        cls_logits_list, bbox_regs_list = self.head(features)

        feature_shapes = [(f.shape[2], f.shape[3]) for f in features]
        anchors_per_level = self.anchor_gen(feature_shapes, device)

        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_sum = torch.tensor(0.0, device=device)
        total_pos = 0

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(device)
            cls_loss_b, reg_sum_b, num_pos_b = self._compute_loss_single_image(
                cls_logits_list,
                bbox_regs_list,
                anchors_per_level,
                gt_boxes,
                feature_shapes,
                b,
                device,
            )
            total_cls_loss = total_cls_loss + cls_loss_b
            total_reg_sum = total_reg_sum + reg_sum_b
            total_pos += num_pos_b

        if total_pos > 0:
            avg_reg_loss = total_reg_sum / float(total_pos)
        else:
            avg_reg_loss = torch.tensor(0.0, device=device)

        total_loss = total_cls_loss + self.lambda_reg * avg_reg_loss

        return {
            "loss_cls": total_cls_loss,
            "loss_reg": avg_reg_loss,
            "loss_total": total_loss,
        }

    # ------------------------------------------------------------------
    # INFERENCE (batched)
    # ------------------------------------------------------------------
    def _forward_inference(self, images):
        device = images[0].device
        x = pad_images_to_batch(images).to(device)
        B = x.shape[0]

        features = self.backbone(x)  # [P3, P4, P5]
        cls_logits_list, bbox_regs_list = self.head(features)
        feature_shapes = [(f.shape[2], f.shape[3]) for f in features]
        anchors_per_level = self.anchor_gen(feature_shapes, device)

        outputs = []
        orig_sizes = [(img.shape[1], img.shape[2]) for img in images]

        for b in range(B):
            all_boxes = []
            all_scores = []

            for cls_l, box_l, anchors_l in zip(cls_logits_list, bbox_regs_list, anchors_per_level):
                cls_l_img = cls_l[b]   # (A*C, H, W)
                box_l_img = box_l[b]   # (A*4, H, W)

                A_C, Hf, Wf = cls_l_img.shape
                A = self.num_anchors
                N_l = Hf * Wf * A

                cls_flat = cls_l_img.permute(1, 2, 0).reshape(N_l, self.num_classes)
                box_flat = box_l_img.permute(1, 2, 0).reshape(N_l, 4)

                scores = torch.sigmoid(cls_flat[:, 0])  # single foreground class
                keep = scores > 0.05
                if keep.sum() == 0:
                    continue

                scores_k = scores[keep]
                box_k = box_flat[keep]
                anchors_k = anchors_l[keep]

                decoded = decode_boxes(anchors_k, box_k)
                all_boxes.append(decoded)
                all_scores.append(scores_k)

            if len(all_boxes) == 0:
                outputs.append({
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                })
                continue

            boxes_cat = torch.cat(all_boxes, dim=0)
            scores_cat = torch.cat(all_scores, dim=0)

            # clip boxes to original image size
            h, w = orig_sizes[b]
            boxes_cat[:, 0] = boxes_cat[:, 0].clamp(min=0, max=w - 1)
            boxes_cat[:, 2] = boxes_cat[:, 2].clamp(min=0, max=w - 1)
            boxes_cat[:, 1] = boxes_cat[:, 1].clamp(min=0, max=h - 1)
            boxes_cat[:, 3] = boxes_cat[:, 3].clamp(min=0, max=h - 1)

            keep_nms = nms(boxes_cat, scores_cat, 0.5)
            boxes_cat = boxes_cat[keep_nms]
            scores_cat = scores_cat[keep_nms]
            labels_cat = torch.ones_like(scores_cat, dtype=torch.int64, device=device)

            outputs.append({
                "boxes": boxes_cat,
                "scores": scores_cat,
                "labels": labels_cat,
            })

        return outputs


# ======================================================================================
# TRAINING LOOP (LOCAL RETINANET)
# ======================================================================================

def train_one_epoch_local_retina(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]

        loss_dict = model(images, targets)
        loss = loss_dict["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(
                f"[Local-RetinaNet][Epoch {epoch}] Step {step} "
                f"cls: {loss_dict['loss_cls'].item():.4f}, "
                f"reg: {loss_dict['loss_reg'].item():.4f}, "
                f"total: {loss.item():.4f}"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"{timestamp()} — Local RetinaNet Epoch {epoch} Avg Loss: {total_loss / len(loader):.4f}")


# ======================================================================================
# COCO EVAL (shared for both models)
# ======================================================================================

def convert_to_coco_predictions(outputs, image_ids):
    results = []
    for out, img_id in zip(outputs, image_ids):
        for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    return results


def evaluate_coco(model, loader, ann_file, device, out_name):
    fix_coco_json(ann_file)
    print(f"Evaluating COCO mAP → {out_name} ...")
    model.eval()
    coco = COCO(ann_file)
    results = []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="COCO Eval"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            image_ids = [t["image_id"].item() for t in targets]
            results.extend(convert_to_coco_predictions(outputs, image_ids))

    ensure_dir("./output/coco_eval")
    pred_file = f"./output/coco_eval/{out_name}"
    with open(pred_file, "w") as f:
        json.dump(results, f)

    if len(results) == 0:
        print("No predictions generated — skipping COCOeval (AP will be 0).")
        return

    coco_dt = coco.loadRes(pred_file)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()


# ======================================================================================
# LOCAL RETINANET INFERENCE HELPERS
# ======================================================================================

def run_single_inference_local(model_path, image_path, config):
    device = config.device
    print(f"Loading Local RetinaNet: {model_path}")

    # Load model
    model = LocalRetinaNet(
        num_classes=config.num_classes_local,
        lambda_reg=config.lambda_reg_local,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError("Image not found: " + image_path)

    orig_h, orig_w = img_bgr.shape[:2]

    # Convert to PIL for ResizeForDetection
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    resize = ResizeForDetection(max_side=1024)

    # Fake target (ResizeForDetection expects a dict)
    fake_target = {"boxes": torch.zeros((0,4))}

    resized_img_tensor, _ = resize(img_pil, fake_target)  # tensor CHW

    resized_h, resized_w = resized_img_tensor.shape[1], resized_img_tensor.shape[2]

    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    # Move to device
    tensor = resized_img_tensor.to(device)

    with torch.no_grad():
        outputs = model([tensor])
        out = outputs[0]

    boxes_scaled = out["boxes"].clone()
    boxes_scaled[:, 0] *= scale_x   # x1
    boxes_scaled[:, 2] *= scale_x   # x2
    boxes_scaled[:, 1] *= scale_y   # y1
    boxes_scaled[:, 3] *= scale_y   # y2

    vis = img_bgr.copy()
    for box, score in zip(boxes_scaled, out["scores"]):
        if score < 0.24:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def batch_inference_local(model_path, folder, config,
                          save_dir="./output/inference_retina_local"):
    device = config.device
    print("\n=== Batch Inference: Local RetinaNet ===")
    ensure_dir(save_dir)

    model = LocalRetinaNet(
        num_classes=config.num_classes_local,
        lambda_reg=config.lambda_reg_local,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    image_files = [f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    with torch.no_grad():
        for file in tqdm(image_files, desc="Local RetinaNet Batch Infer"):
            path_img = os.path.join(folder, file)
            img_bgr = cv2.imread(path_img)
            if img_bgr is None:
                print("Skipping unreadable image:", file)
                continue

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = F.to_tensor(rgb).to(device)

            outputs = model([tensor])
            out_det = outputs[0]

            vis = img_bgr.copy()
            for box, score in zip(out_det["boxes"], out_det["scores"]):
                if score < 0.3:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            save_path = os.path.join(save_dir, f"{Path(file).stem}_retina_local.jpg")
            cv2.imwrite(save_path, vis)

    print("\n=== Local RetinaNet Batch Inference Completed ===")


class ResizeForDetection:
    def __init__(self, max_side=1024, stride=32):
        self.max_side = max_side
        self.stride = stride  # ensure divisibility for FPN

    def __call__(self, img, target):
        # img is PIL
        w, h = img.size
        scale = self.max_side / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Make divisible by FPN stride
            new_w = (new_w // self.stride) * self.stride
            new_h = (new_h // self.stride) * self.stride

            img = img.resize((new_w, new_h))

            # Resize boxes too:
            boxes = target["boxes"]
            boxes = boxes * scale
            target["boxes"] = boxes

        img = F.to_tensor(img)
        return img, target

# ======================================================================================
# MAIN MENU
# ======================================================================================

def main():
    print("\n========== Dense Detection Launcher ==========")
    print("LOCAL RETINANET (Correlation-Aware Focal)")
    print(" 1. Train Local RetinaNet")
    print(" 2. Evaluate Local RetinaNet (COCO mAP)")
    print(" 3. Single-image inference (Local RetinaNet)")
    print(" 4. Batch inference (Local RetinaNet)")
    print("----------------------------------------------")
    print("RETINANET BASELINE (torchvision)")
    print(" 5. Train RetinaNet baseline")
    print(" 6. Evaluate RetinaNet baseline (COCO mAP)")
    print(" 7. Single-image inference (RetinaNet baseline)")
    print(" 8. Batch inference (RetinaNet baseline)")
    print("==============================================")

    choice = input("Enter option (1-8): ").strip()
    if choice not in [str(i) for i in range(1, 9)]:
        print("Invalid choice.")
        return

    device = config.device
    print("Using device:", device)

    # ---------------- Local RetinaNet ----------------
    if choice == "1":
        ensure_dir("./output/pt-models")
        train_ds = SKU110K_COCO(
            config.train_images,
            config.train_annotations,
            transforms=ResizeForDetection(max_side=1024),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size_local,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
        model = LocalRetinaNet(
            num_classes=config.num_classes_local,
            lambda_reg=config.lambda_reg_local,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        for ep in range(1, config.num_epochs_local + 1):
            train_one_epoch_local_retina(model, train_loader, optimizer, device, ep)

        torch.save(model.state_dict(), config.save_local_model_path)
        print("Saved Local RetinaNet model:", config.save_local_model_path)
        return

    if choice == "2":
        test_ds = SKU110K_COCO(
            config.test_images,
            config.test_annotations,
            transforms=ResizeForDetection(max_side=1024),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size_local,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

        model = LocalRetinaNet(
            num_classes=config.num_classes_local,
            lambda_reg=config.lambda_reg_local,
        ).to(device)
        model.load_state_dict(torch.load(config.save_local_model_path,
                                         map_location=device))

        evaluate_coco(model, test_loader, config.test_annotations, device,
                      out_name="pred_retinanet_local.json")
        return

    if choice == "3":
        run_single_inference_local(
            config.save_local_model_path, config.infer_image_path, config
        )
        return

    if choice == "4":
        batch_inference_local(
            config.save_local_model_path, config.test_images, config
        )
        return

    # ---------------- RetinaNet Baseline ----------------
    if choice == "5":
        ensure_dir("./output/pt-models")
        train_ds = SKU110K_COCO(
            config.train_images,
            config.train_annotations,
            transforms=None,   # torchvision model handles raw tensor images
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size_retina,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

        model = create_retinanet(config.num_classes_retina).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        for ep in range(1, config.num_epochs_retina + 1):
            train_one_epoch_retinanet(model, train_loader, optimizer, device, ep)

        torch.save(model.state_dict(), config.save_retinanet_model_path)
        print("Saved RetinaNet baseline:", config.save_retinanet_model_path)
        return

    if choice == "6":
        test_ds = SKU110K_COCO(
            config.test_images,
            config.test_annotations,
            transforms=None,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size_retina,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

        model = create_retinanet(config.num_classes_retina).to(device)
        model.load_state_dict(torch.load(config.save_retinanet_model_path,
                                         map_location=device))

        evaluate_coco(model, test_loader, config.test_annotations, device,
                      out_name="pred_retinanet.json")
        return

    if choice == "7":
        run_inference_retinanet(
            config.save_retinanet_model_path, config.infer_image_path, config, save_output=False
        )
        return

    if choice == "8":
        batch_inference_retinanet(
            config.save_retinanet_model_path, config.test_images, config
        )
        return


if __name__ == "__main__":
    main()