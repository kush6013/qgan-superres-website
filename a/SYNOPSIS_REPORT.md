# Project Synopsis Report

## Project Title
**QGAN-Based Medical Image Super-Resolution and Secure Web Gallery**

## TABLE OF CONTENTS
| S. No. | Section | Page No. |
|---|---|---|
| 1 | Broad Area of Work | 1 |
| 2 | Introduction | 2 |
| 3 | Literature Survey / Work done in the field of proposed work | 5 |
| 4 | Existing Gaps | 11 |
| 5 | Objectives of the proposed work | 14 |
| 6 | Proposed Methodologies | 16 |
| 7 | Expected Outcome of the proposed work | 22 |
| 8 | Future scope of the work | 23 |
| 9 | References | 25 |

---

## 1. Broad Area of Work
This project belongs to the broad area of:
- Medical Image Processing
- Artificial Intelligence and Deep Learning
- Generative Adversarial Networks (GANs) and Quantum-Inspired GAN concepts
- Full-stack Healthcare AI Application Development

The work combines medical scan enhancement, machine learning model integration, secure user workflow, and persistent result management through a deployable web platform.

---

## 2. Introduction
Medical imaging is central to diagnosis and treatment planning, but many scans are captured at low resolution or contain degraded visual details. In practice, low-quality scans can reduce visibility of fine boundaries, textures, and subtle structures that clinicians rely on.

This project proposes and implements a **QGAN-inspired medical image super-resolution system** with a complete web interface and backend pipeline. Instead of building only a research script, this project provides an end-to-end workflow:
- User registration/login with session-based authentication
- Medical image upload and validation
- Super-resolution enhancement using a GAN-based generator
- Result storage in database and searchable gallery
- Admin dashboard for users and upload moderation

### 2.1 Problem Statement
There is a gap between research-grade image enhancement models and practical systems usable by real users. Most projects stop at model training; they do not provide secure upload, metadata, user access control, and persistent gallery operations.

### 2.2 Project Scope
The implemented system supports:
- Input formats: PNG, JPG, JPEG, DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`)
- Automatic conversion of medical volumes (DICOM/NIfTI) to preview images
- 8x upscaling workflow for super-resolution output
- Metadata capture: description, tags, patient ID
- Per-user result ownership (`owner_id`) and search filters

### 2.3 Implemented System Modules (Exact from Project)
1. **Web Framework and UI**
   - FastAPI backend (`main.py`)
   - Jinja2 templates (`templates/`)
   - Static JS upload flow (`static/js/main.js`)
2. **Authentication and User Management**
   - Registration, login, logout, profile update
   - Password reset token generation and expiry
   - Role-based admin access (`is_admin`)
3. **Medical Validation Pipeline**
   - Custom binary classifier (ResNet18-based) when trained checkpoint exists
   - Fallback ImageNet heuristic + grayscale/saturation checks
4. **Super-Resolution Pipeline**
   - QGAN generator inference if checkpoint is available
   - Bicubic fallback if generator checkpoint is missing
5. **Database and Persistence**
   - SQLAlchemy with SQLite (`database.db`)
   - Two core tables: `users`, `results`
6. **Result Management**
   - Gallery display, search, download, and delete features
   - API endpoints for result listing and lookup

---

## 3. Literature Survey / Work done in the field of proposed work
Recent work in this area is moving in two parallel directions:
- Strong classical GAN-based medical image enhancement and synthesis
- Quantum-inspired or hybrid quantum-classical generative learning

The present project is best positioned as a **QGAN-inspired practical system** with real deployment features, while advanced quantum training and formal benchmark evaluation remain future extension points.

### 3.1 Key Literature Themes
1. **Quantum-inspired medical generation**
   - Work such as *Quantum Generative Learning for High-Resolution Medical Image Generation* (Khatun et al., 2025) motivates variational quantum + classical adversarial designs.
2. **Hybrid quantum-classical GAN architecture research**
   - Studies report potential representational advantages, but practical deployment complexity remains high.
3. **Classical medical GAN benchmarking**
   - Comparative studies show strong classical baselines still dominate in practical settings.
4. **Medical super-resolution trends**
   - Newer methods use attention and multi-scale fusion; many include PSNR/SSIM/FID evaluation pipelines.

### 3.2 Relevance to This Project
This project adopts the research motivation of QGAN-like enhancement while prioritizing deployability:
- End-to-end upload-to-result workflow
- Secure user sessions and administration
- Database-backed history and searchable gallery
- Medical format support in production-style API routes

### 3.3 Honest Positioning of Current Implementation
The current repository includes:
- Classical CNN-based generator + discriminator
- Optional PennyLane-based refinement block in model architecture
- Practical web deployment pipeline

The current repository does **not yet** include:
- Full variational quantum circuit training pipeline integrated in production training
- Formal PSNR/SSIM/FID experiment notebook/report pipeline
- Clinical-grade validation against curated benchmark datasets

---

## 4. Existing Gaps
Based on literature and current implementation status, the following gaps are identified:

1. **Benchmarking gap**
   - No complete automated metric suite (PSNR, SSIM, FID) is integrated in the repository’s training/evaluation workflow.
2. **Quantum implementation gap**
   - The architecture is QGAN-inspired with optional hybrid block, but large-scale validated quantum training is not yet completed.
3. **Clinical validation gap**
   - No formal radiologist-in-the-loop evaluation is available in current project artifacts.
4. **Dataset standardization gap**
   - Training scripts exist, but public benchmark-level dataset protocol and split reporting are not yet documented in thesis-ready detail.
5. **Security hardening gap**
   - The app has authentication/session support, but enterprise-level controls (RBAC granularity, audit trails, encryption-at-rest, compliance logging) are future work.
6. **Strict medical rejection policy gap**
   - The upload pipeline performs medical-likeness detection, but currently does not hard-reject all uncertain files; it allows supported formats and warns on low confidence.

---

## 5. Objectives of the Proposed Work
The objectives of this project are:

1. Build a practical AI system for medical image super-resolution using a QGAN-inspired model.
2. Support real-world medical upload formats including DICOM and NIfTI.
3. Provide secure user management with session login, profile control, and reset-password flow.
4. Maintain a structured result repository with metadata, ownership, and searchable gallery.
5. Provide an admin panel for monitoring users and uploaded outputs.
6. Enable model training scripts for both super-resolution model and medical/non-medical classifier.
7. Prepare the system for future formal benchmarking and quantum-hybrid experimentation.

---

## 6. Proposed Methodologies
This project uses a combined methodology across AI modeling, data processing, backend engineering, and web application design.

### 6.1 System Architecture
The architecture has four layers:
1. **Presentation layer**: Jinja2 templates + JavaScript drag-and-drop upload UI.
2. **Application layer**: FastAPI routes for auth, upload, gallery, admin, and APIs.
3. **AI layer**: Medical image classifier + QGAN-inspired super-resolution generator.
4. **Persistence layer**: SQLite with SQLAlchemy ORM for users and results.

### 6.2 Data and File Handling Method
1. User uploads image/scan file.
2. Extension and content type checks are applied.
3. If DICOM/NIfTI, preview image is generated through normalization and slice extraction.
4. Medical classifier evaluates whether upload resembles medical scan.
5. Super-resolution executes:
   - QGAN generator inference if checkpoint available (`models/qgan_generator.pt`)
   - Bicubic resize fallback otherwise
6. Result image is saved in `results/`.
7. Metadata + ownership are stored in `results` table.

### 6.3 Database Methodology
The project uses SQLAlchemy ORM with migration-style column checks in `database.py`.

Core entities:
- `users`: identity, password hash + salt, admin flag, reset token info, created time
- `results`: file paths, metadata fields, owner foreign key, timestamp

Relationship:
- One user can own multiple results (`users.id` -> `results.owner_id`)

(ER diagram image created in project: `ER_DIAGRAM.png`)

### 6.4 Model Methodology
1. **Generator**
   - Residual CNN blocks and PixelShuffle upsampling (8x output)
   - Optional `QuantumRefinementBlock` if PennyLane is enabled
2. **Discriminator**
   - Convolutional classifier for adversarial realism feedback
3. **Loss design in training**
   - Adversarial loss (BCE)
   - Content loss (L1)
   - Weighted generator objective: `g_loss = g_content + 1e-3 * g_adv`

### 6.5 Classifier Methodology
1. Custom binary ResNet18 classifier (medical vs non-medical) when checkpoint exists.
2. Fallback heuristic uses ImageNet predictions + keyword rejection + color/grayscale heuristics.
3. This improves practical filtering in open-upload environments.

### 6.6 Authentication and Security Methodology
1. Passwords are hashed with SHA-256 + per-user random salt.
2. Session middleware controls login state.
3. Admin route is protected by `is_admin` check.
4. Password reset uses one-time token with expiry window.

### 6.7 Implementation Stack
- FastAPI, Jinja2, SQLAlchemy
- PyTorch, Torchvision, NumPy, Pillow
- Optional medical IO libraries: `pydicom`, `nibabel`
- Optional quantum library: `pennylane`
- Uvicorn runtime

### 6.8 Training Workflow
1. QGAN training script (`qgan_train.py`):
   - Uses HR image folder `data/hr`
   - Builds LR-HR pairs using bicubic downscale/upscale
   - Trains generator + discriminator adversarially
2. Medical classifier training script (`medical_classifier_train.py`):
   - Uses folder split:
     - `data/medical/train/medical`
     - `data/medical/train/nonmedical`
   - Optional validation directory
   - Saves classifier checkpoint to `models/medical_classifier.pt`

---

## 7. Expected Outcome of the Proposed Work
Expected outcomes of this project are:

1. A functional web application that enhances uploaded medical scans to higher resolution.
2. Support for standard image and medical formats (PNG/JPG/JPEG/DICOM/NIfTI).
3. Persistent, searchable gallery with metadata and user ownership.
4. Secure user workflow with authentication and admin visibility.
5. Reusable training scripts for model improvement and retraining.
6. A QGAN-inspired baseline platform that can be upgraded to deeper hybrid quantum-classical experimentation.

---

## 8. Future scope of the work
Future extensions can include:

1. Formal quantitative evaluation dashboard with PSNR, SSIM, FID reporting.
2. Stronger quantum-classical integration with end-to-end hybrid training experiments.
3. Multi-modality optimization for MRI/CT/X-ray specific preprocessing pipelines.
4. Better clinical validation with expert radiologist feedback and task-based scoring.
5. Security hardening for production deployment (audit log, encryption policy, compliance layer).
6. Containerized deployment and cloud inference scaling.
7. Integration of explainability overlays for enhanced image interpretability.
8. Rich admin analytics (usage trends, model confidence summaries, dataset drift alerts).

---

## 9. References
1. Khatun, A., Aydeniz, K. Y., Weinstein, Y. S., Usman, M. (2025). *Quantum generative learning for high-resolution medical image generation*. Machine Learning: Science and Technology, 6, 025032.
2. QTML 2024. *Quantum Generative Learning for High-Resolution Medical Image Generation* (conference abstract).
3. Abdulqader, D. M., Abdulazeez, A. M. (2025). *A Comparative Study of Generative Adversarial Networks in Medical Image Processing*.
4. *Quantum generative adversarial network for image generation* (Springer, 2025).
5. Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NeurIPS.
6. Ledig, C., et al. (2017). *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. CVPR.
7. Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE TIP.
8. Kingma, D. P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.

### Project Implementation References (Codebase)
9. `main.py` (FastAPI app, upload/generation/auth/gallery routes).
10. `models/qgan_model.py` (generator/discriminator and optional quantum refinement block).
11. `models/medical_classifier.py` (medical scan validation pipeline).
12. `qgan_train.py` and `medical_classifier_train.py` (training scripts).
13. `database.py`, `models/user.py`, `models/result.py` (database schema and ORM).

