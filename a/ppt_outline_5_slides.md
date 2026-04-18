# 5-Slide PPT Outline

## Slide 1: Title and Problem
**Title:** QGAN-Inspired Medical Image Super-Resolution and Secure Gallery System

**Points to include:**
- Medical images may suffer from low resolution and loss of detail
- Existing research often focuses on models, not usable end-to-end systems
- This project builds a practical web prototype for upload, validation, 8x enhancement, and result management

**Speaker note:**
This project focuses on improving medical image quality while also solving the practical problem of how users upload, manage, and access enhanced results through a web application.

## Slide 2: Literature Direction
**Title:** Research Background

**Points to include:**
- QIGL (IOP 2025): quantum generator + classical discriminator for medical image generation
- HybridQ / MediQ-GAN: newer hybrid or quantum-inspired medical image generation direction
- MDPI comparative GAN study: Pix2Pix, SPADE, and WGAN show that evaluation depends on task
- Recent classical medical SR papers use attention, multi-scale fusion, and strong benchmarking

**Speaker note:**
The literature shows two main directions: advanced classical medical GANs and emerging quantum-inspired or hybrid quantum-classical models. My project is motivated by both, but implemented in a practical classical way.

## Slide 3: Paper vs Project Comparison
**Title:** Where My Project Stands

**Points to include:**
- Papers are stronger in model novelty and benchmark evaluation
- My project is stronger in end-to-end usability and deployment workflow
- Current repo is QGAN-inspired, not quantum-implemented
- Current repo includes authentication, upload validation, database storage, gallery, and admin workflow

**Suggested table summary:**
- `Research papers`: advanced architectures, FID/PSNR/SSIM benchmarking, limited deployment
- `My project`: simpler classical model, no full benchmarking yet, strong web application workflow

**Speaker note:**
This is the most honest way to position the project. It is not claiming to outperform recent quantum papers. Instead, it turns the idea into a usable system.

## Slide 4: System Architecture
**Title:** Proposed System

**Points to include:**
- FastAPI backend
- SQLAlchemy + SQLite database
- Medical image validation before enhancement
- Classical residual generator with pixel-shuffle upsampling
- 8x super-resolution with adversarial training
- Result storage and searchable gallery

**Workflow:**
1. User logs in
2. Uploads scan
3. Validation checks if image is medical
4. Generator enhances image at 8x scale
5. Result is stored and shown in gallery

**Speaker note:**
The project is designed as an end-to-end pipeline, not just a model training script.

## Slide 5: Results, Limitations, and Future Scope
**Title:** Outcome and Future Work

**Points to include:**
- Working prototype for upload, enhancement, and result management
- Medical-image filtering and gallery workflow implemented
- Current limitations:
  - no true quantum module
  - current upload support is PNG, JPG, and JPEG only
  - no integrated PSNR/SSIM/FID benchmarking
  - simpler architecture than recent medical SR papers
- Future scope:
  - benchmark against bicubic and classical GAN baselines
  - add PSNR, SSIM, and FID
  - explore PennyLane or Qiskit-based hybrid extensions

**Speaker note:**
The current work is a strong prototype foundation. The next academic step is rigorous benchmarking and, if feasible, hybrid quantum-classical experimentation.
