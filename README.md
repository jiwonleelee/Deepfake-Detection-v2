# Deepfake Detection v2: Advanced Generalization Study

> **Note**: This project is an individual development focused on enhancing the robustness and generalization of deepfake detection models.
> 🔗 **Looking for the initial team project?** Check out [Deepfake-Detection-v1](https://github.com/jiwonleelee/Deepfake-Detection-v1)

---

## 🎯 Project Goals
The primary goal of this project is to overcome the **generalization gap** in deepfake detection. While current models perform well on seen datasets, they often fail against unseen manipulation techniques. This project investigates a holistic approach combining data synthesis, architectural modification, and temporal analysis.

## 🚀 Key Features & Development Plan

### 1. Data Augmentation (Pre-processing)
* **Goal**: Learn universal manipulation artifacts rather than dataset-specific noise.
* **Method**: Implementation of **Self-Blended Images (SBI)** to synthesize training samples on-the-fly, forcing the model to focus on blending boundaries.

### 2. Architectural Evolution (Spatial Analysis)
* **Backbone**: ConvNeXt-Tiny (Modernized CNN).
* **Innovation**: Moving beyond simple branching to **Multi-scale Feature Fusion** or **Frequency-domain Branches** (based on CVPR/ICCV research) to capture fine-grained texture inconsistencies.

### 3. Temporal Integration (Sequence Analysis)
* **Approach**: A **Hierarchical Structure** that stacks a temporal module (GRU or Transformer) on top of the frozen spatial backbone.
* **Focus**: Analyzing facial muscle dynamics and jittering artifacts across frames to detect sophisticated forgeries.

---

## 📅 Roadmap (In Progress)

- [x] Phase 1: Baseline setup with ConvNeXt-Tiny (Team Project baseline)
- [ ] Phase 2: Integration of SBI (Self-Blended Images) pipeline
- [ ] Phase 3: Research-driven architectural modifications & Ablation studies
- [ ] Phase 4: Implementation of Temporal Head (Transformer vs. GRU)
- [ ] Phase 5: Cross-dataset evaluation (Celeb-DF v2, DFDC, DeeperForensics)

---

## 📚 References
* *Self-Blended Images for Generalizable Deepfake Detection (CVPR 2022)*
* *A ConvNet for the 2020s (CVPR 2022)*
* *M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection (2022)*
* *Video Vision Transformer (ViViT) (ICCV 2021)*
