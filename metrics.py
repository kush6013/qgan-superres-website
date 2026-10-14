import numpy as np
from PIL import Image


def _to_grayscale_array(img: Image.Image) -> np.ndarray:
    """Converts a PIL image to a 2D float32 grayscale numpy array in range [0, 255]."""
    gray = img.convert("L")
    return np.asarray(gray, dtype=np.float32)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Higher is generally closer to baseline fidelity.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 1e-10:
        return 100.0
    max_pixel = 255.0
    psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    return float(round(psnr, 2))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two single-channel images.
    Returns a score between -1 and 1 (typically 0.70 - 0.99 for good reconstructions).
    """
    c1 = (k1 * 255.0) ** 2
    c2 = (k2 * 255.0) ** 2

    # Block-based mean and variances
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim = numerator / (denominator + 1e-10)
    return float(round(np.clip(ssim, -1.0, 1.0), 4))


def calculate_laplacian_sharpness(img_gray: np.ndarray) -> float:
    """
    Computes edge sharpness using variance of the discrete Laplacian operator.
    Higher values represent sharper edges and richer high-frequency structural detail.
    """
    if img_gray.shape[0] < 3 or img_gray.shape[1] < 3:
        return 0.0

    # 3x3 discrete Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(img_gray, 1, mode="reflect")
    h, w = img_gray.shape

    # Fast 2D convolution for small kernel
    laplacian = (
        padded[:-2, 1:-1] +
        padded[2:, 1:-1] +
        padded[1:-1, :-2] +
        padded[1:-1, 2:] -
        4 * padded[1:-1, 1:-1]
    )

    sharpness_score = float(np.var(laplacian))
    return float(round(sharpness_score, 2))


def evaluate_super_resolution_quality(
    original_img: Image.Image,
    enhanced_img: Image.Image,
    ground_truth_img: Image.Image | None = None,
) -> dict:
    """
    Evaluates comprehensive Image Quality Assessment (IQA) metrics:
    - PSNR (dB)
    - SSIM index
    - Edge Sharpness (Laplacian variance)
    - Sharpness Detail Gain (%) over standard Bicubic interpolation
    - Reference evaluation type (bicubic baseline vs ground truth)

    Note: When ground truth is unavailable, PSNR and SSIM are computed
    against the bicubic upscaled input baseline. This measures fidelity
    relative to interpolation, NOT proven clinical reconstruction accuracy.
    """
    orig_w, orig_h = original_img.size
    enh_w, enh_h = enhanced_img.size

    # Handle reference comparison target
    if ground_truth_img is not None:
        ref_img = ground_truth_img.resize((enh_w, enh_h), Image.LANCZOS)
        reference_type = "ground_truth"
        reference_description = "Evaluated against paired high-resolution ground truth image"
    else:
        ref_img = original_img.resize((enh_w, enh_h), Image.BICUBIC)
        reference_type = "bicubic_baseline"
        reference_description = "Evaluated against bicubic-upscaled input baseline (measures relative fidelity, not ground-truth clinical accuracy)"

    # Baseline comparison for sharpness gain
    bicubic_baseline = original_img.resize((enh_w, enh_h), Image.BICUBIC)

    arr_enh = _to_grayscale_array(enhanced_img)
    arr_ref = _to_grayscale_array(ref_img)
    arr_bicubic = _to_grayscale_array(bicubic_baseline)

    psnr_score = calculate_psnr(arr_enh, arr_ref)
    ssim_score = calculate_ssim(arr_enh, arr_ref)

    enh_sharpness = calculate_laplacian_sharpness(arr_enh)
    bicubic_sharpness = calculate_laplacian_sharpness(arr_bicubic)

    # Sharpness detail gain percentage over bicubic
    if bicubic_sharpness > 1e-5:
        sharpness_gain_pct = round(((enh_sharpness - bicubic_sharpness) / bicubic_sharpness) * 100.0, 1)
    else:
        sharpness_gain_pct = 0.0

    scale_factor = round(enh_w / max(1, orig_w), 1)

    return {
        "psnr": psnr_score,
        "ssim": ssim_score,
        "sharpness": enh_sharpness,
        "bicubic_sharpness": bicubic_sharpness,
        "sharpness_gain_pct": sharpness_gain_pct,
        "input_resolution": f"{orig_w}x{orig_h}",
        "output_resolution": f"{enh_w}x{enh_h}",
        "scale_factor": f"{scale_factor}x",
        "reference_type": reference_type,
        "reference_description": reference_description,
        "clinical_disclaimer": "Quantitative metrics reflect algorithmic image similarity and do not constitute clinical diagnostic proof.",
    }
