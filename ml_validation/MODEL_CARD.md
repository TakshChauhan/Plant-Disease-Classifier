# Model Card — Plant Disease Classifier (ResNet18, 38-Class PlantVillage)

## 1. Model Details & Architecture

| Property | Verified Specification |
| :--- | :--- |
| **Repository** | [TakshChauhan/Plant-Disease-Classifier](https://github.com/TakshChauhan/Plant-Disease-Classifier) |
| **Checkpoint Path** | `results/best_model.pt` |
| **Checkpoint SHA-256** | `f456fe4cd6042b2c031172bcdab00a40c845475a3f71fa052264d28311ed0cc9` |
| **Checkpoint File Size** | `44,865,035` bytes (`42.7866 MiB`) |
| **Checkpoint Format** | Legacy v1 dict (`model_state`, `model_name`, `class_names`); loads safely with `weights_only=True` |
| **Backbone Architecture** | `torchvision.models.resnet18` (`BasicBlock` [2, 2, 2, 2]) |
| **Classification Head** | `nn.Linear(in_features=512, out_features=38, bias=True)` |
| **Total Learnable Parameters** | `11,196,006` (`11,176,512` backbone + `19,494` classification head across 102 parameter tensors) |
| **Non-Learnable Buffers** | `9,620` elements across 20 `BatchNorm2d` layers (`4,800` `running_mean` + `4,800` `running_var` + `20` `num_batches_tracked`) |
| **Theoretical Complexity** | `1,813,580,800` MACs (`1.8136 GMACs` / `~3.6272 GFLOPs`) per `3×224×224` forward pass |
| **Evaluation Preprocessing** | EXIF transpose → Resize shorter side to `255` (`int(224 * 1.14)`, bilinear) → `CenterCrop(224)` → ImageNet normalize (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) |
| **Explainability Target** | `model.layer4[-1]` (`layer4.1` output tensor of shape `(B, 512, 7, 7)`) via Grad-CAM ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)) |

## 2. Supported Classes (38 Classes Across 14 Crop Species)

The checkpoint classifies inputs into exactly 38 classes ordered lexicographically (matching `torchvision.datasets.ImageFolder`):

| Index | Raw Class Identifier | Crop Species | Condition / Disease | Etiology Category | FC Weight L2 Norm | FC Bias |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| 0 | `Apple___Apple_scab` | Apple | Apple scab (*Venturia inaequalis*) | Fungal | 2.7905 | +0.0226 |
| 1 | `Apple___Black_rot` | Apple | Black rot (*Botryosphaeria obtusa*) | Fungal | 3.0052 | +0.0623 |
| 2 | `Apple___Cedar_apple_rust` | Apple | Cedar apple rust (*Gymnosporangium juniperi-virginianae*) | Fungal | 2.9272 | +0.0337 |
| 3 | `Apple___healthy` | Apple | Healthy | Healthy | 2.8453 | +0.0568 |
| 4 | `Blueberry___healthy` | Blueberry | Healthy | Healthy | 3.0067 | -0.0585 |
| 5 | `Cherry_(including_sour)___Powdery_mildew` | Cherry | Powdery mildew (*Podosphaera clandestina*) | Fungal | 3.1379 | +0.0189 |
| 6 | `Cherry_(including_sour)___healthy` | Cherry | Healthy | Healthy | 3.1180 | -0.1149 |
| 7 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Corn (maize) | Gray leaf spot (*Cercospora zeae-maydis*) | Fungal | 2.7285 | +0.0224 |
| 8 | `Corn_(maize)___Common_rust_` | Corn (maize) | Common rust (*Puccinia sorghi*) | Fungal | 3.1605 | +0.0122 |
| 9 | `Corn_(maize)___Northern_Leaf_Blight` | Corn (maize) | Northern Leaf Blight (*Exserohilum turcicum*) | Fungal | 2.6743 | +0.0277 |
| 10 | `Corn_(maize)___healthy` | Corn (maize) | Healthy | Healthy | 3.4039 | +0.1040 |
| 11 | `Grape___Black_rot` | Grape | Black rot (*Guignardia bidwellii*) | Fungal | 3.0389 | +0.0029 |
| 12 | `Grape___Esca_(Black_Measles)` | Grape | Esca / Black Measles (*Phaeomoniella* spp.) | Fungal | 2.9482 | -0.0569 |
| 13 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Grape | Isariopsis Leaf Spot (*Pseudocercospora vitis*) | Fungal | 3.2948 | +0.0637 |
| 14 | `Grape___healthy` | Grape | Healthy | Healthy | 3.2517 | +0.0588 |
| 15 | `Orange___Haunglongbing_(Citrus_greening)` | Orange | Huanglongbing (*Candidatus* Liberibacter spp.) | Bacterial | 3.2902 | -0.0149 |
| 16 | `Peach___Bacterial_spot` | Peach | Bacterial spot (*Xanthomonas arboricola*) | Bacterial | 3.0598 | +0.0306 |
| 17 | `Peach___healthy` | Peach | Healthy | Healthy | 3.1449 | -0.0026 |
| 18 | `Pepper,_bell___Bacterial_spot` | Pepper, bell | Bacterial spot (*Xanthomonas campestris*) | Bacterial | 2.7429 | -0.0645 |
| 19 | `Pepper,_bell___healthy` | Pepper, bell | Healthy | Healthy | 2.8388 | -0.0286 |
| 20 | `Potato___Early_blight` | Potato | Early blight (*Alternaria solani*) | Fungal | 2.9806 | -0.0262 |
| 21 | `Potato___Late_blight` | Potato | Late blight (*Phytophthora infestans*) | Oomycete | 2.7805 | -0.0275 |
| 22 | `Potato___healthy` | Potato | Healthy | Healthy | 2.7967 | +0.0283 |
| 23 | `Raspberry___healthy` | Raspberry | Healthy | Healthy | 3.0017 | -0.0048 |
| 24 | `Soybean___healthy` | Soybean | Healthy | Healthy | 3.0819 | +0.0614 |
| 25 | `Squash___Powdery_mildew` | Squash | Powdery mildew (*Podosphaera xanthii*) | Fungal | 3.2693 | +0.0449 |
| 26 | `Strawberry___Leaf_scorch` | Strawberry | Leaf scorch (*Diplocarpon earliana*) | Fungal | 3.1862 | +0.0364 |
| 27 | `Strawberry___healthy` | Strawberry | Healthy | Healthy | 2.9598 | -0.0312 |
| 28 | `Tomato___Bacterial_spot` | Tomato | Bacterial spot (*Xanthomonas* spp.) | Bacterial | 2.7586 | -0.0383 |
| 29 | `Tomato___Early_blight` | Tomato | Early blight (*Alternaria solani*) | Fungal | 2.2768 | +0.0373 |
| 30 | `Tomato___Late_blight` | Tomato | Late blight (*Phytophthora infestans*) | Oomycete | 2.3096 | -0.0656 |
| 31 | `Tomato___Leaf_Mold` | Tomato | Leaf Mold (*Passalora fulva*) | Fungal | 2.6968 | -0.0432 |
| 32 | `Tomato___Septoria_leaf_spot` | Tomato | Septoria leaf spot (*Septoria lycopersici*) | Fungal | 2.3953 | -0.0346 |
| 33 | `Tomato___Spider_mites Two-spotted_spider_mite` | Tomato | Two-spotted spider mite (*Tetranychus urticae*) | Arachnid / Pest | 2.5235 | -0.0528 |
| 34 | `Tomato___Target_Spot` | Tomato | Target Spot (*Corynespora cassiicola*) | Fungal | 2.2280 | +0.0133 |
| 35 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomato | Tomato Yellow Leaf Curl Virus (TYLCV) | Viral | 3.2248 | -0.0534 |
| 36 | `Tomato___Tomato_mosaic_virus` | Tomato | Tomato mosaic virus (ToMV) | Viral | 3.0870 | +0.1225 |
| 37 | `Tomato___healthy` | Tomato | Healthy | Healthy | 3.1686 | +0.0379 |

## 3. Dataset Provenance & Licensing

1. **Primary Training & Validation Source**: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle (`70,295` training images, `17,572` validation images, `33` unlabeled test images at `256×256` RGB).
2. **Underlying Primary Collection**: [PlantVillage Dataset](https://arxiv.org/abs/1511.08060) ([Hughes & Salathé, 2015](https://arxiv.org/abs/1511.08060); [Mohanty et al., 2016, *Frontiers in Plant Science*](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full)), consisting of ~54,305 laboratory photographs of single detached leaves placed on uniform studio paper backdrops under controlled illumination.
3. **External Field Benchmark Target**: [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) ([Singh et al., 2020, *ACM CoDS COMAD*](https://doi.org/10.1145/3371158.3371196), licensed under **CC BY 4.0**; see also [Dataset Ninja — PlantDoc](https://datasetninja.com/plantdoc)), comprising ~2,598 real-world field/in-situ images across 13 crop species and 27–29 classes.

## 4. Training Methodology & Forensic Verification

Forensic inspection of `results/best_model.pt` (`ml_validation/evaluate_baseline.py`) established the exact training recipe:

- **Two-Stage Transfer Learning Schedule**:
  - **Phase 1 (Frozen Backbone)**: 3 epochs training only `model.fc` with Adam (`lr = 1e-3`, `batch_size = 32`).
  - **Phase 2 (Full Fine-Tuning)**: 7 epochs training all `11,196,006` parameters with Adam (`lr = 1e-4`, `batch_size = 32`).
- **Forensic BatchNorm Proof**: Every one of the 20 `BatchNorm2d` layers in `best_model.pt` has `num_batches_tracked == 21,970`. Because `ceil(70,295 / 32) = 2,197` batches/epoch, `21,970 = 10 × 2,197` proves that:
  1. The checkpoint was saved at the conclusion of Epoch 10 (index 9 in `results/training_curves.png`).
  2. Early stopping did not fire in either Phase 1 or Phase 2.
  3. During Phase 1, `model.train()` was called without `set_batchnorm_eval()`, so BatchNorm running means and variances updated during all 3 frozen-head epochs as well as all 7 fine-tuning epochs.

## 5. Evaluation Metrics: Reported Claims vs. Independently Verified Evidence

| Metric / Property | Original README Claim | Independent Verification Status | Verified Value / Finding |
| :--- | :--- | :--- | :--- |
| **PlantVillage `valid/` Top-1 Accuracy** | `99.35%` (`n=17,572`) | **UNREPRODUCED (Dataset Not Bundled)** | Blocked locally because the 1.3 GB Kaggle dataset is not stored in git; `evaluate_baseline.py` is ready to run on `data/valid`. |
| **PlantVillage `valid/` Macro P / R / F1** | `99.3%` / `99.3%` / `99.3%` | **UNREPRODUCED (Dataset Not Bundled)** | Subject to offline-augmentation train/valid leakage and checkpoint-selection optimism. |
| **Checkpoint Integrity & Determinism** | `best_model.pt` (~45 MB) | **VERIFIED** | `42.79 MiB`, SHA-256 `f456fe4c...`, deterministic outputs (`atol < 1e-6`), valid probability simplex. |
| **CPU Single-Image Latency (`B=1`)** | Not measured in README | **VERIFIED (`benchmark.py`)** | Median `56.80 ms` (`mean 58.63 ms`, `p90 59.73 ms`, `p95 72.65 ms`, `p99 86.13 ms`; throughput `17.61 img/s`). |
| **CPU Forward + Grad-CAM Latency (`B=1`)** | Not measured in README | **VERIFIED (`benchmark.py`)** | Median `57.98 ms` (`mean 60.35 ms`, `p95 80.57 ms`). |
| **Non-Leaf OOD Overconfidence (`n=150`)** | Anecdotal check (`78%` on 1 noise image) | **VERIFIED (`calibration.py`)** | Mean Top-1 confidence `81.82%` (median `95.93%`) across 150 non-leaf images; `100%` of uniform/Gaussian noise images exceed `0.95` confidence (`Corn (maize) — Healthy`). |
| **Background-Only Shortcut Probe** | Not measured in README | **VERIFIED (`robustness.py`)** | Pure soil texture predicted as `Orange — Huanglongbing` (`95.51%`); pure white backdrop as `Apple — Cedar apple rust` (`88.72%`); pure black backdrop as `Tomato — Healthy` (`82.89%`). |
| **Background Replacement Instability** | Not measured in README | **VERIFIED (`robustness.py`)** | Replacing uniform studio background with black fill (`87.5%`), wood/bark (`81.25%`), canopy clutter (`56.25%`), or soil (`37.5%`) flips Top-1 predictions on procedural leaf specimens. |

## 6. Intended Use & Out-of-Scope Use

- **Intended Use**:
  - Educational demonstration of PyTorch transfer learning, two-phase fine-tuning, Grad-CAM spatial attribution, and ML validation methodology on laboratory-style single-leaf photographs.
  - Benchmark baseline for research into shortcut learning, dataset leakage, and domain adaptation between laboratory (`PlantVillage`) and field (`PlantDoc`) agricultural imagery.
- **Out-of-Scope Use (Strictly Prohibited)**:
  - **Commercial agricultural diagnosis, fungicide/pesticide prescription, crop culling, or quarantine decisions** (especially for regulated pathogens such as *Candidatus* Liberibacter / Citrus Greening or *Phytophthora infestans*).
  - Unconstrained field photography containing multiple leaves, soil/sky/canopy backgrounds, fruit, stems, whole plants, nutrient deficiencies, insect damage not in the 38 classes, or crops outside the 14 supported species.

## 7. Known Limitations, Generalization Uncertainty & Ethical Risks

1. **Train/Validation Augmentation & Specimen Leakage**: The Kaggle source dataset applied offline data augmentation before splitting into `train/` and `valid/`, and the underlying PlantVillage dataset includes multiple photographs per physical plant without public specimen IDs. In-distribution validation accuracy (`99.35%`) overstates true generalization even on laboratory leaves.
2. **Severe Domain Shift on Field Images**: Prior literature ([Mohanty et al., 2016](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full); [Singh et al., 2020](https://doi.org/10.1145/3371158.3371196)) documents that PlantVillage-trained CNNs drop to 20%–35% accuracy on natural field images due to background reliance. Our Phase 4 background-replacement and background-only probes confirm that `best_model.pt` strongly couples background color/texture to disease predictions.
3. **Extreme Out-of-Distribution Overconfidence**: Because `Corn_(maize)___healthy` has the largest FC weight L2 norm (`3.4039`) and second-highest positive bias (`+0.1040`), high-variance noise and unfamiliar inputs systematically collapse onto `Corn_(maize)___healthy` or `Tomato___healthy` with >90% softmax probability. A fixed probability threshold (such as `0.60`) rejects only `21.3%` of non-leaf OOD inputs (`78.7%` false acceptance rate).
4. **Center-Crop Blindness to Leaf Tips and Margins**: The `Resize(255) -> CenterCrop(224)` pipeline discards ~22.8% of the area of a square image and >60% of elongated rectangular smartphone photos (`16:9` or `2:1`), blinding the classifier to early tip burn or marginal necrosis.
