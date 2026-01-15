# Paper Scripts

Scripts used for generating results and figures for the AlveolEye paper. Each script can be run as a module from the project root.

## Dataset

The `optimal_training_size.py` script requires a training dataset. You can download it from Google Drive using the `--download-dataset` flag or by calling the utility function directly:

```python
from alveoleye.paper_scripts._utils import download_training_dataset
dataset_path = download_training_dataset()
```

The other scripts (`confidence_maps.py`, `trials.py`, `save_snapshots.py`) only require input images for inference and will use the default model weights unless a custom weights path is specified.

## Scripts

### optimal_training_size.py

Determines the optimal number of training images by running incremental training experiments until a target validation loss threshold is met.

```bash
python -m alveoleye.paper_scripts.optimal_training_size [dataset_path] [options]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `dataset_path` | str | None | Path to dataset directory. Can be omitted if `--download-dataset` is used. |
| `--download-dataset` | flag | - | Download the training dataset from Google Drive before running. |
| `--threshold` | float | 0.5 | Target validation loss threshold to achieve. |
| `--start-images` | int | 50 | Initial number of images to train with. |
| `--step-size` | int | 50 | Number of images to add for each subsequent run. |
| `--max-images` | int | None | Maximum number of images to try (default: all available). |
| `--epochs` | int | 100 | Number of training epochs per run. |
| `--device` | str | auto | Device to train on. Choices: `auto`, `cuda`, `cpu`, `mps`. |
| `--seed` | int | 42 | Random seed for reproducibility. |
| `--output-dir` | str | None | Directory to save results CSV and checkpoints. |
| `--save-checkpoints` | flag | - | Save model checkpoints for each run (requires `--output-dir`). |

**Examples:**

```bash
# Download dataset and run with defaults
python -m alveoleye.paper_scripts.optimal_training_size --download-dataset

# Use local dataset with custom parameters
python -m alveoleye.paper_scripts.optimal_training_size /path/to/dataset \
    --threshold 0.3 --step-size 25 --epochs 50 --output-dir ./results
```

---

### confidence_maps.py

Generates confidence heatmaps for model predictions on a directory of images. Produces per-class heatmaps and combined overlay images.

```bash
python -m alveoleye.paper_scripts.confidence_maps [options]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | str | `../../example_images` | Path to directory containing input images. |
| `--output-dir` | str | **required** | Directory to save output heatmap images. |
| `--weights-path` | str | `../../default_weights/default.pth` | Path to model weights file. |
| `--colorbar-orientation` | str | vertical | Orientation of the colorbar. Choices: `vertical`, `horizontal`. |

**Examples:**

```bash
# Generate heatmaps with default model
python -m alveoleye.paper_scripts.confidence_maps \
    --input-dir ./my_images \
    --output-dir ./heatmaps

# Use custom weights
python -m alveoleye.paper_scripts.confidence_maps \
    --input-dir ./my_images \
    --output-dir ./heatmaps \
    --weights-path ./my_model.pth
```

---

### trials.py

Runs experimental trials to evaluate model behavior: determinism testing, random line location sensitivity, and variable line quantity analysis.

```bash
python -m alveoleye.paper_scripts.trials <trial> [options]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `trial` | str | **required** | Trial to run. Choices: `determinism_trial` (or `1`), `random_line_location_trial` (or `2`), `variable_line_quantity_trial` (or `3`). |
| `--input-dir` | str | `../../example_images` | Path to directory containing input images. |
| `--iterations` | int | 15 | Number of iterations per image (also max lines for variable_line_quantity trial). |
| `--output-dir` | str | None | Export location for results CSV. Required for trials 2 and 3. |
| `--weights-path` | str | None | Path to model weights file (uses default if not specified). |

**Trial Types:**

- **determinism_trial (1)**: Tests if the model produces consistent results across multiple runs on the same images.
- **random_line_location_trial (2)**: Evaluates sensitivity to randomized line placement during assessment.
- **variable_line_quantity_trial (3)**: Analyzes how results vary with different numbers of measurement lines.

**Examples:**

```bash
# Run determinism trial (output optional)
python -m alveoleye.paper_scripts.trials determinism_trial

# Run random line location trial with output
python -m alveoleye.paper_scripts.trials 2 \
    --input-dir ./images \
    --output-dir ./results \
    --iterations 20

# Run variable line quantity trial
python -m alveoleye.paper_scripts.trials variable_line_quantity_trial \
    --input-dir ./images \
    --output-dir ./results \
    --iterations 100
```

---

### save_snapshots.py

Generates and saves intermediate images from each stage of the AlveolEye processing pipeline for a single input image.

```bash
python -m alveoleye.paper_scripts.save_snapshots [options]
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-image` | str | `../../example_images/10.png` | Path to the input image. |
| `--output-dir` | str | **required** | Directory to save intermediate snapshot images. |
| `--weights-path` | str | None | Path to model weights file (uses default if not specified). |

**Examples:**

```bash
# Generate snapshots for default example image
python -m alveoleye.paper_scripts.save_snapshots --output-dir ./snapshots

# Generate snapshots for custom image
python -m alveoleye.paper_scripts.save_snapshots \
    --input-image ./my_image.png \
    --output-dir ./snapshots
```

---

## Output Formats

- **optimal_training_size.py**: CSV file with columns: `n_images`, `best_val_loss`, `final_epoch`, `training_time_seconds`, `meets_threshold`
- **confidence_maps.py**: PNG heatmap images organized by input image name
- **trials.py**: CSV file with trial-specific metrics
- **save_snapshots.py**: PNG images of each pipeline stage
