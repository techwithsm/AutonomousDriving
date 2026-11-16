# Autonomous Driving – AIML Capstone

Two-part project:

| Part | Objective | Data | Approach |
|---|---|---|---|
| **1** | Predict the **type of vehicle** in an image and **localise it with a bounding box** | `data/Part 1/Images/Images/*.jpg` + `labels.csv` (5,626 traffic-camera images, 17,967 boxes, 11 classes) | YOLO11 CNN object detector (COCO-pretrained, fine-tuned) |
| **2** | Analyse autopilot usage and its effect on road safety | `data/Part 2/Tesla - Deaths.csv` (294 fatal events) | Data inspection, cleaning, exploratory data analysis (EDA) |

---

## Project structure

```
AutonomousDriving/
├── configs/
│   ├── part1_detection.yaml      # paths, classes, split, model, training, eval, inference
│   └── part2_tesla.yaml          # input, outputs, columns to drop, cleaning rules
├── src/autodrive/
│   ├── common/                   # config loader (+CLI overrides), logging, seed, plot style
│   ├── detection/                # PART 1
│   │   ├── workspace.py          #   Task 1  parent folder + child folders
│   │   ├── dataset.py            #   Task 2  unzip, clean labels, stratified split, YOLO export
│   │   ├── eda.py                #           class balance, box sizes, ground-truth samples
│   │   ├── model.py              #   Task 3  architecture choice + training
│   │   ├── evaluate.py           #   Task 4  test metrics (mAP, P, R, per-class AP)
│   │   └── inference.py          #   Task 5  predictions, GT comparison, vehicle counts
│   └── tesla/                    # PART 2
│       ├── cleaning.py           #   Task 1  inspection (dtypes, missing, duplicates) + cleaning
│       ├── analysis.py           #   Task 2  a–d: every EDA question as a pure function
│       ├── plots.py              #           figures
│       └── pipeline.py           #           end-to-end run -> tables, figures, report.md
├── scripts/
│   ├── run_part1.py              # CLI: setup | prepare | eda | train | evaluate | predict | all
│   └── run_part2.py              # CLI: full Part 2 pipeline
├── notebooks/
│   ├── Part1_Vehicle_Detection.ipynb
│   └── Part2_Tesla_EDA.ipynb     # executed, with all outputs
├── tests/                        # pytest unit tests for both parts
├── data/                         # input data (git-ignored)
├── workspace/                    # Part 1 training workspace (generated, git-ignored)
└── outputs/                      # Part 2 results (generated, git-ignored)
```

## Setup

Python 3.10–3.13 is recommended (Python 3.14 isn't yet supported by every ML wheel).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
# CPU-only PyTorch (use the CUDA index URL from pytorch.org if you have an NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
pytest -q
```

**Pretrained weights behind a proxy or SSL inspection.** If Ultralytics fails with
`SSLCertVerificationError` while downloading `yolo11*.pt`, download the file another way
(browser, or `Invoke-WebRequest`) into `workspace/weights/`. The code uses a local copy there automatically:

```powershell
Invoke-WebRequest https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt -OutFile workspace\weights\yolo11s.pt
```

---

## Part 1 – Vehicle detection

```powershell
python scripts/run_part1.py all                 # every stage
python scripts/run_part1.py setup               # Task 1
python scripts/run_part1.py prepare             # Task 2
python scripts/run_part1.py eda                 # dataset plots
python scripts/run_part1.py train               # Task 3
python scripts/run_part1.py evaluate            # Task 4
python scripts/run_part1.py predict             # Task 5
# overrides – any config key:
python scripts/run_part1.py train --set train.epochs=10 train.fraction=0.25 train.device=cpu
python scripts/run_part1.py predict --set predict.source="C:/path/to/new/images"
```

### Task 1 – Folder layout
```
workspace/vehicle_detection/        <- parent folder for custom model training
├── raw/                            <- zip extraction target
├── dataset/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/    <- YOLO .txt labels
│   ├── data.yaml
│   └── split_manifest.csv
├── runs/                           <- weights, curves, confusion matrices
├── inference/                      <- annotated predictions, counts, GT comparison
└── reports/                        <- EDA plots, metrics JSON/CSV
```

### Task 2 – Dataset preparation
* **Unzipping.** If `Images.zip` is supplied instead of a folder, only image members are extracted.
  Extraction is parallel (`extract_workers` threads), skips the `__MACOSX` junk folder, and resumes where it
  left off, so an interrupted extraction on a slow VM continues instead of restarting.
* **Labels.** `labels.csv` describes 110,000 images; only the 17,967 boxes of the 5,626 images we
  have are kept. Boxes are clipped to the image, degenerate boxes are dropped, and duplicates are removed.
* **Split.** 80/10/10 at image level, **stratified on the rarest class in each image**, so even
  motorcycle (84 images) appears in train, val and test (see `reports/split_summary.csv`).
* **Format.** YOLO labels (`class cx cy w h`, normalised per image, so both 720×480 and 342×228
  images work). Images are **hard-linked**, which takes no time and no extra disk space.

### Task 3 – Architecture choice: YOLO11 (CNN, one-stage, anchor-free)

| Candidate | Speed | Small objects | Multiple objects per image | Verdict |
|---|---|---|---|---|
| Custom CNN + box-regression head | fast | poor | ✗ (one box) | images contain 1–27 vehicles |
| Faster R-CNN (two-stage) | slow | good | ✓ | too slow for real-time traffic |
| SSD | fast | weak | ✓ | anchor-based, misses small vehicles |
| **YOLO11** | **real-time** | **good** | ✓ | **chosen** |

The choice follows from what the data showed:
* Half of all boxes cover less than 0.8% of the image. YOLO11 detects at 3 scales (PAN-FPN, strides 8/16/32).
* The dataset is small (4,500 training images), so **COCO-pretrained** weights are used. COCO already includes
  car, bus, truck, person, bicycle and motorcycle.
* Classes are imbalanced (cars are 67% of boxes). Mosaic and mixup augmentation help, and the
  anchor-free decoupled head is used.
* Cameras film by day and by night, so HSV colour augmentation is used. Images are never flipped vertically
  (roads are never upside down).

Network: Conv-BN-SiLU stem → C3k2 CSP blocks → SPPF → C2PSA attention → PAN neck →
decoupled classification and box heads (DFL + CIoU loss). The model size is set with `model.weights`:
`yolo11n.pt` (fast), `yolo11s.pt` (default) or `yolo11m.pt` (more accurate).

**Compute.** The default 60 epochs at 640 px is slow on CPU. Options:
* Run on Google Colab or a GPU VM (`train.device=0`); the first cell of the notebook covers Colab.
* Or, on CPU, `--set model.weights=yolo11n.pt train.imgsz=480 train.epochs=20`.

### Task 4 – Evaluation
`evaluate` runs on the **held-out test split** and writes:
* `reports/metrics_test.json`: precision, recall, F1, mAP@0.5, mAP@0.5:0.95 and speed
* `reports/per_class_metrics_test.csv` and `per_class_ap_test.png`: AP for each class, alongside its number of test boxes
* `runs/<run>_test/`: confusion matrix and PR/F1 curves from Ultralytics

### Task 5 – Inference
`predict` writes:
* `inference/annotated/`: images with predicted boxes
* `inference/gt_vs_pred.png`: ground truth and prediction side by side
* `inference/sample_accuracy.csv`: true positives (TP), false positives (FP) and missed boxes (FN) for each image, matched by class at IoU ≥ 0.5
* `inference/vehicle_counts.csv`: vehicles of each type per image (the vehicle-counting use case)

---

## Part 2 – Tesla autopilot & road safety

```powershell
python scripts/run_part2.py
```
Outputs in `outputs/part2/`: `tesla_deaths_clean.csv`, `tables/*.csv` (23 tables),
`figures/*.png` (14 figures) and **`report.md`**, which contains every answer, figure and table.
`notebooks/Part2_Tesla_EDA.ipynb` walks through the same steps interactively.

| PDF task | Where it is answered |
|---|---|
| 1a dtypes, missing values, duplicates | `cleaning.inspect`, `cleaning.duplicates_report` |
| 1b remove irrelevant columns | `configs/part2_tesla.yaml → cleaning.drop_columns` |
| 2a events by date / year / day, per state & country | `events_by_year/month/date/weekday`, `country_year_matrix`, `state_year_matrix`, `country_weekday_matrix`, `state_weekday_matrix`, `events_per_day_rate` |
| 2b victims per accident | `deaths_per_event` |
| 2b times Tesla drivers died | `tesla_driver_deaths` |
| 2b proportion of events with occupant deaths | `occupant_death_proportion` |
| 2b cyclist/pedestrian events | `cyclist_ped_distribution` |
| 2b Tesla driver/occupant + cyclist/pedestrian deaths | `tesla_and_cyclist_ped` |
| 2b collisions with other vehicles | `other_vehicle_frequency` |
| 2c events across models | `model_distribution`, `model_year_matrix` |
| 2d verified autopilot deaths | `verified_autopilot` |

**Cleaning decisions**
* 13 rows without a case number were removed: totals, stray numbers and a footnote.
* Links, source, note and victim-name columns were dropped.
* `-` was read as 0 in count columns, and as Unknown or Non-US in Model and State.
* Year typo `202` was corrected to 2020.
* Where Year and Date disagreed (5 cases), the year was taken from Date; the original is kept as `year_reported`.
* "Holland" was merged into "Netherlands".
* Invalid model values `1` and `2` were set to Unknown.

**Headline results**
* 294 fatal events and 353 deaths, from 2013 to 2023.
* Events rose from 2 in 2013 to 93 in 2022.
* 73% of events were in the USA; California alone has 42% of US events.
* A Tesla driver died in 117 events (40%); a driver or occupant died in 47%.
* A cyclist or pedestrian died in 44 events (15%); another vehicle was involved in 37%.
* **19 verified autopilot deaths in 16 events (5% of events).** Autopilot was claimed in 35 events.
* *Limitation:* there is no miles-driven figure, so the data shows how many deaths occurred, not risk per mile.
