# QGAN Training Data

This folder is where you put the high-resolution training images used to train the QGAN model.

## Structure

- `data/hr/`
  - Put your raw high-resolution images here.
  - Supported formats: `PNG`, `JPG`, `JPEG`.
  - The training script will load all images recursively from this folder.

## Image recommendations

- Use clean, high-quality medical scans.
- Prefer images with consistent lighting and minimal compression artifacts.
- Crop or resize images to a square format if possible, e.g. `256x256` or `512x512`.
- The script currently uses a default high-resolution crop size of `256`.

## How it works

The training script generates a low-resolution input image by downsampling each high-resolution image by the scale factor (default `8x`). It then trains the QGAN generator to reconstruct the original high-resolution image.

## Training command

From the project root:

```bash
./venv/bin/python qgan_train.py --data-dir data/hr --epochs 20 --batch-size 8
```

If you have a GPU, leave `CUDA` enabled in the environment. If not, training will fall back to CPU.

## Parameter tuning suggestions

- `--batch-size`:
  - `8` is a good default for mid-range GPUs.
  - Use `4` if memory is limited.
  - Use `16` or `32` if you have a large GPU and still get stable training.
- `--lr`:
  - Start with `1e-4`.
  - If the loss jumps or the model is unstable, try `5e-5`.
- `--hr-size`:
  - `256` is a good default crop size.
  - Use `192` if your images are smaller or memory is tight.
  - Use `320` or `384` only if your GPU can handle it and your data is high-resolution.
- `--epochs`:
  - `20` is a safe starting point.
  - Increase to `40` or `80` for larger datasets.
- Augmentation:
  - Use `--augment` to enable random flips, rotations, and color jitter.
  - For medical scans, keep `--rotation` small, e.g. `10` degrees.
  - Set jitter values low, e.g. `--brightness 0.1 --contrast 0.1 --saturation 0.1 --hue 0.05`.

## Quality recommendations

- Use clean, high-resolution scans.
- Avoid heavy JPEG artifacts and over-compressed images.
- Keep the dataset consistent in modality (e.g. all chest X-rays or all CT slices).
- Prefer images with good contrast and clear edges.
