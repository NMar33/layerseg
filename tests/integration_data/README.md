# Integration Test Reference Data

## Purpose

These files allow integration tests to verify that the binarizer pipeline produces **exactly the same output** after code changes (model re-save, smart_contrast optimization, refactoring, etc.).

The idea: we run the pipeline once, save its output as "expected", and then every future test run compares fresh pipeline output against these saved files.

## How It Works

### Test flow

```
1. pytest runs tests/test_integration.py
2. Pipeline processes 5 small images from inputs/ (128x128, ~54 sec total)
3. Tests compare pipeline output against saved files in expected/
4. If output matches → PASS (code change didn't break anything)
   If output differs → FAIL (code change altered pipeline behavior)
```

### What is compared

| Test class | Compares | Tolerance | Purpose |
|-----------|---------|-----------|---------|
| `TestSoftMaskMatchesReference` | Soft probability mask (grayscale) | ≤1 pixel value diff | Catches changes in model inference or post-processing |
| `TestThresholdMaskMatchesReference` | Binary mask at t=0.5 and t=0.7 | Bit-exact (zero diff) | Catches any change in thresholding logic |
| `TestIoUAboveMinimum` | Pipeline output vs ground truth | IoU > 0.05 | Catches catastrophic quality degradation |

### Pipeline settings used in tests

- `scale_factors: [1]` (no scaling — fast)
- `gaussian_blur: false`
- `binarizer_thresholds: [0.5, 0.7]`
- `color_interest: "black"`
- `device: "cpu"`

## Directory Structure

```
integration_data/
├── inputs/                              # 5 test images, 128x128 PNG
│   ├── real_crop.png                    # Crop from real image (16_1.tif)
│   ├── synth_type02_crop.png            # Crop from test_dataset synthetic
│   ├── wavy_thin_gray_bg_128x128.png    # Generated wavy lines
│   ├── horiz_uniform_128x128.png        # Generated straight lines
│   └── checker_uniform_128x128.png      # Generated checkerboard
│
├── expected/                            # Reference pipeline output (15 files)
│   ├── {name}_soft.png                  # Soft mask (grayscale 0-255)
│   ├── {name}_t0.5.png                  # Binary mask at threshold 0.5
│   └── {name}_t0.7.png                  # Binary mask at threshold 0.7
│   ... (× 5 images = 15 files)
│
├── ground_truth/                        # GT masks for IoU check (5 files)
│   └── {name}.png                       # Binary: white=foreground, black=background
│
├── generate_reference.py                # Script to regenerate expected/
└── README.md                            # This file
```

**All files are tracked in git.** Tests work immediately after `git clone` + dependency install.

## Running Tests

```bash
cd project_phasian_binarizer_dev_v00

# Integration tests only (~54 sec)
python -m pytest tests/test_integration.py -v

# Unit tests only (~10 sec)
python -m pytest tests/ -v --ignore=tests/test_integration.py

# All tests (~64 sec)
python -m pytest tests/ -v
```

## When Tests Fail After a Code Change

### Scenario 1: Unintentional breakage

You changed something and tests fail unexpectedly. The change broke the pipeline — investigate and fix.

### Scenario 2: Intentional behavior change

You deliberately changed something that alters output (examples below). In this case:

1. **Verify visually** that the new output is correct
2. **Regenerate reference data:**
   ```bash
   python tests/integration_data/generate_reference.py
   ```
   Takes ~60 seconds. Overwrites files in `expected/`.
3. **Re-run tests** to confirm they pass with new reference
4. **Commit** the updated reference files along with your code changes

### Changes that require regeneration

| Change | Why output changes |
|--------|-------------------|
| Model re-save (joblib → state_dict) | Float precision may differ slightly |
| smart_contrast optimization (Python loops → vectorized) | Floating point order of operations |
| PyTorch version upgrade | Internal numerics may differ |
| Post-processing logic change | Direct output change |
| Threshold/mask inversion logic | Direct output change |

### Changes that should NOT require regeneration

| Change | Why output stays the same |
|--------|--------------------------|
| Refactoring report generation | Reports don't affect masks |
| Adding new CLI options | Doesn't change pipeline core |
| Changing logging | No effect on output |
| Adding new image formats | Existing inputs unchanged |

## Regenerating Reference Data

### Prerequisites

Reference generation needs source images from assessment. Run at least once:
```bash
python -m assessment.generate
python -m assessment.prepare_test_dataset
```

### Generate

```bash
python tests/integration_data/generate_reference.py
```

This script:
1. Copies/crops 5 images from assessment data into `inputs/`
2. Runs the full binarizer pipeline (scale=1, no blur, thresholds 0.5+0.7)
3. Extracts soft and binary masks into `expected/`
4. Copies ground truth masks into `ground_truth/`

### Verify

```bash
python -m pytest tests/test_integration.py -v
# Should show 20 passed
```

## Mask Conventions

**Pipeline output masks** (in `expected/`):
- Detected foreground (lines) = **BLACK (0)**
- Background = **WHITE (255)**
- This is because `color_interest="black"` inverts the mask during save

**Ground truth masks** (in `ground_truth/`):
- Foreground (lines) = **WHITE (255)**
- Background = **BLACK (0)**

The IoU test accounts for this inversion (compares dark pixels in predicted vs white pixels in GT).
