# AI Research Benchmarking & Ground-Truth Evaluation Methodology

This document outlines the theoretical principles, statistical assumptions, and algorithmic evaluation standards governing the **QGAN Medical Image Super-Resolution Research Benchmarking Framework**.

---

## 1. What is a Benchmark?

In medical image processing and computational vision, an evaluation benchmark is a **controlled, reproducible scientific protocol** designed to compare multiple reconstruction algorithms under identical conditions. A valid benchmark guarantees:
- **Identical Input Distribution**: All algorithms receive the exact same degraded pixel observations.
- **Identical Preprocessing & Postprocessing**: Normalization, spatial resampling, and bit-depth policies are standardized.
- **Identical Hardware Telemetry**: Inference latency (ms) and peak VRAM allocation (MB) are profiled under equivalent hardware constraints.
- **Unbiased Ground-Truth Verification**: Metrics are computed against an uncorrupted high-resolution anatomical reference.

A benchmark does not prove clinical efficacy; rather, it quantifies mathematical fidelity, perceptual similarity, and computational efficiency under controlled constraints.

---

## 2. What is Real Ground Truth? (Mode A)

**Real Paired Ground Truth** refers to prospective clinical or physical phantom acquisitions where both low-resolution (LR) and high-resolution (HR) volumes represent genuine physical scanner data of the same patient anatomy:
- **Acquisition Protocol**: The HR volume is acquired with standard clinical high-resolution protocols (e.g., thin-slice 0.5 mm CT, high-field 3T MRI, or high-mAs radiography), while the LR volume is acquired with accelerated protocols (e.g., thick-slice 5.0 mm CT, k-space undersampled MRI, or low-dose radiography).
- **Spatial Registration**: Real paired data requires rigid or non-rigid motion correction and co-registration to account for patient movement, breathing, and scanner shifts between acquisitions.
- **Reference Label**: Records evaluated in this mode carry `reference_type: "ground_truth"`.

---

## 3. What is Synthetic Ground Truth? (Mode B)

**Synthetic Ground Truth** refers to the paradigm where high-resolution clinical images serve as the reference, and low-resolution inputs are generated through an explicit mathematical degradation operator:
$$y = \mathcal{D}(x; \theta) + \eta$$
where $x$ is the pristine high-resolution source image, $\mathcal{D}$ represents spatial downsampling and optical/sensor point spread function (PSF) blur, and $\eta$ denotes stochastic noise (e.g., Gaussian electronic noise or Poisson quantum mottle).

- **Role**: Allows benchmarking when prospective paired clinical acquisitions are unavailable or unethical (e.g., avoiding double radiation exposure to patients).
- **Reference Label**: Records evaluated in this mode carry `reference_type: "synthetic_ground_truth"`.
- **Methodological Distinction**: Synthetic ground-truth experiments evaluate an algorithm's ability to **invert a known mathematical downsampling operator**, which is not equivalent to reconstructing genuine clinical scanner physics.

---

## 4. What Does PSNR Mean Here?

**Peak Signal-to-Noise Ratio (PSNR)** is an objective engineering metric measuring logarithmic fidelity based on Mean Squared Error (MSE):
$$\text{MSE} = \frac{1}{M \times N} \sum_{i=1}^M \sum_{j=1}^N \left( \hat{I}(i, j) - I_{\text{ref}}(i, j) \right)^2$$
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) \quad (\text{dB})$$
where $\text{MAX}_I = 255.0$ for 8-bit normalized representations.

### Technical Meaning & Limitations:
- **Pixel-Level Fidelity**: Higher PSNR indicates that reconstructed pixels are numerically close to reference pixels.
- **Smoothing Bias**: PSNR heavily penalizes high-frequency noise and slight phase shifts. Consequently, models optimized purely for L2/MSE loss often generate over-smoothed, blurry reconstructions that achieve high PSNR but lack diagnostic structural clarity.
- **Non-Clinical**: A 1.5 dB improvement in PSNR does not signify that microcalcifications, subtle fractures, or vascular stenoses are more diagnostically visible.

---

## 5. What Does SSIM Mean Here?

**Structural Similarity Index (SSIM)** measures visual degradation by evaluating changes in structural information, luminance, and contrast:
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
where $\mu$ represents local window intensity means, $\sigma^2$ represents local variances, and $\sigma_{xy}$ is cross-covariance.

### Technical Meaning & Limitations:
- SSIM scores range between $-1.0$ and $+1.0$ (typically $0.70$ to $0.98$ for super-resolved radiology images).
- It reflects preservation of anatomical edges, tissue interfaces, and organ contours better than pure MSE.
- However, like PSNR, SSIM evaluates local statistical correlations and cannot verify whether high-frequency textures generated by GANs are authentic anatomical reality or synthetic hallucinations.

---

## 6. What Does the Bicubic Baseline Represent?

The **Bicubic Interpolation Baseline** represents standard polynomial signal resampling without learned semantic priors:
- It serves as the **mandatory lower-bound reference anchor** for all comparative benchmarks.
- Any deep learning model (classical CNN, GAN, or Quantum-Classical Hybrid) must demonstrably outperform bicubic interpolation ($\Delta\text{PSNR} > 0$, $\Delta\text{SSIM} > 0$) to justify its computational overhead and model complexity.
- If a deep model scores lower than bicubic interpolation on a subset of samples ($\Delta\text{PSNR} < 0$), this signals **destructive hallucination, high-frequency artifacts, or severe domain distribution shift**.

---

## 7. Limitations of Synthetic Degradation

While synthetic degradation provides deterministic control, its clinical limitations must be explicitly recognized:
1. **Oversimplified Sensor Physics**: Real medical imaging modalities (e.g., helical CT or gradient-echo MRI) exhibit non-isotropic point spread functions, patient involuntary motion, beam hardening, partial volume averaging, and complex non-Cartesian k-space trajectories that simple bicubic downsampling cannot capture.
2. **Noise Heterogeneity**: Real CT exhibits spatially correlated Poisson-Gaussian quantum mottle influenced by dose modulation, whereas synthetic noise is typically modeled as stationary additive Gaussian noise.
3. **Inverse Crime Risk**: If the training degradation operator and the benchmark degradation operator are identical, neural networks may simply learn to invert that specific mathematical kernel without generalizing to genuine scanner physics.

---

## 8. Why Patient-Level Splitting Matters (Leakage Hazards)

In clinical imaging, single examinations frequently generate dozens or hundreds of contiguous 2D slices (e.g., CT or volumetric MRI series) for a single patient.

### The Leakage Hazard:
If dataset splitting is performed randomly at the **individual slice/image level**:
- Slice $k$ from Patient A may land in the **training set**.
- Slice $k+1$ from Patient A may land in the **test set**.

Because contiguous slices share identical patient body habitus, organ geometries, pathology markers, scanner calibration, and noise profiles, the neural network can memorize patient-specific characteristics rather than learning generalizable super-resolution mappings. This results in **falsely inflated test metrics** and catastrophic failure during real-world out-of-distribution evaluation.

### The Solution:
The benchmark framework enforces **Patient-Level Grouping**:
All scans and slices belonging to a given `patient_id` or anonymized subject are assigned strictly and indivisibly to **either** the train, validation, or test split.

---

## 9. Why Benchmark Results Do Not Establish Clinical Diagnostic Benefit

Super-resolution evaluation metrics (PSNR, SSIM, LPIPS, Laplacian Sharpness) are **mathematical signal-processing metrics**. They establish algorithmic convergence, not clinical diagnostic utility.

### Critical Regulatory & Clinical Principles:
1. **Hallucination Risk**: Generative adversarial models (GANs) and diffusion models synthesize high-frequency detail to satisfy perceptual loss functions. In medical imaging, an unconstrained generative model may fabricate realistic-looking trabecular bone patterns, microvessels, or parenchymal textures that do not exist in the patient's body, or conversely, smooth away faint pathological lesions.
2. **Diagnostic Equivalency vs Super-Resolution**: Achieving higher visual sharpness does not prove that a radiologist can detect lung nodules or cerebral ischemia with higher sensitivity or specificity.
3. **Clinical Evaluation Requirement**: Establishing genuine clinical efficacy requires formal reader studies (e.g., ROC/FROC analysis with certified radiologists, lesion detection tasks, and diagnostic concordance trials) adhering to FDA/CE Software as a Medical Device (SaMD) standards.
