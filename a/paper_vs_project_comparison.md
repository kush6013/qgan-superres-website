# Paper vs Project Comparison

## Purpose
This table compares the verified research papers with the current repository so the project can be presented honestly in a synopsis, report, or viva.

| Paper / Study | Main Idea | What It Implements | How It Relates to This Project | Match with Current Repo |
|---|---|---|---|---|
| Khatun et al., 2025, IOP: *Quantum generative learning for high-resolution medical image generation* | Quantum image generative learning for medical images | Variational quantum generator, classical discriminator, PCA-based representation, FID comparison, medical image generation | This is the strongest research motivation for calling the project QGAN-inspired | Partial match: research motivation matches, but the repo does not implement VQCs, PCA, or FID benchmarking |
| QTML 2024 abstract | Early conference version of QIGL | Quantum generator plus classical discriminator for medical image generation | Supports the research direction and literature background | Partial match: conceptually related, not directly implemented |
| Springer 2025: *Quantum generative adversarial network for image generation* | Hybrid QGAN design and evaluation | Hybrid quantum-classical GAN on non-medical image datasets such as handwritten digits and Fashion-MNIST | Useful for explaining QGAN structure and hybrid training ideas | Low match: useful background, but not medical and not super-resolution deployment |
| HybridQ / MediQ-GAN line, 2025 | Hybrid or quantum-inspired medical image generation | Classical-quantum fusion, latent fusion, quantum-inspired generation, medical dataset augmentation | Represents the more advanced direction your project could evolve toward | Partial match: similar theme, but current repo remains classical and focused on enhancement plus web workflow |
| Abdulqader and Abdulazeez, 2025, MDPI | Comparative study of GANs in medical image processing | Pix2Pix, SPADE GAN, WGAN; metrics including PSNR, SSIM, FID, Dice | Strong reference for evaluation expectations and classical baselines | Partial match: same application area, but your repo does not yet include formal benchmark metrics or baseline comparison |
| Scientific Reports 2025: 3D medical image SR with GAN + attention | Advanced medical image super-resolution | 3D modeling, attention, stronger architecture for medical SR | Shows where modern classical medical SR research is heading | Limited match: your repo does super-resolution, but with a simpler 2D residual CNN |
| Electronics 2025: transform-domain multi-scale GAN for medical SR | Improved fidelity for medical super-resolution | Multi-scale fusion, transform-domain design, PSNR/SSIM evaluation | Relevant classical super-resolution benchmark direction | Limited match: same broad task, but your implementation is much simpler |
| arXiv 2025: HQCGAN with transfer learning | Hybrid quantum-classical GAN optimization | Transfer learning analysis for quantum/classical generator-discriminator placement | Useful future-work reference for hybrid model design | Low match today, good future-scope reference |

## Direct Comparison with the Current Repository

| Dimension | Current Repository |
|---|---|
| Core application type | End-to-end FastAPI web app for medical image upload, enhancement, and gallery management |
| Model style | Classical CNN-based generator with residual blocks and pixel-shuffle upsampling |
| Quantum implementation | None in current code |
| Why “QGAN-inspired” still fits | The project motivation and framing come from recent QGAN and quantum-inspired medical image generation literature |
| Medical image validation | Yes, via custom classifier plus fallback heuristics |
| User accounts and gallery | Yes |
| Admin workflow | Yes |
| Benchmark metrics like PSNR, SSIM, FID | Not yet integrated in the current repo |
| Strong baseline comparisons | Not yet implemented |
| Main practical strength | It is a usable prototype, unlike many research papers that focus only on model experiments |
| Main research weakness | It currently lacks rigorous quantitative evaluation and true hybrid quantum components |

## Best Honest Positioning for Viva
The most accurate way to describe the project is:

“This is a QGAN-inspired medical image super-resolution web prototype. The current implementation uses a practical classical generator and medical-image validation workflow, while the project motivation and future scope are guided by recent hybrid quantum-classical medical image generation research.”

## Key Takeaway
Your project is strongest as a deployable prototype. The literature is strongest on advanced model design and benchmarking. That means your best academic positioning is to present the current repo as:

1. stronger than many papers in end-to-end usability,
2. weaker than the latest papers in model novelty and evaluation depth,
3. and well positioned for future extension toward formal benchmarking and hybrid quantum-classical methods.
