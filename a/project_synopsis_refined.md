# Refined Project Synopsis

## Project Title
QGAN-Inspired Medical Image Super-Resolution and Secure Gallery System

## Broad Area of Work
Medical Image Processing, Deep Learning, and AI-Enabled Web Systems

## Abstract
This project presents a medical image super-resolution prototype that combines deep learning with a secure web-based workflow for image upload, enhancement, and result management. The system is designed for medical scan images such as X-rays, CT scans, MRI scans, and related modalities. It includes user authentication, upload validation, model inference, persistent result storage, and gallery browsing.

The repository implements a classical convolutional super-resolution generator inspired by the research direction of quantum generative adversarial networks (QGANs). In the current codebase, the model is not a true quantum or hybrid quantum-classical implementation. Instead, the project uses a practical residual convolutional generator with pixel-shuffle upsampling, a discriminator definition for adversarial training, and a medical-image validation pipeline that filters uploads before enhancement. A FastAPI backend, SQLAlchemy ORM, SQLite database, and Jinja template frontend are used to create an end-to-end application.

The main contribution of this work is not only the image enhancement model, but also the integration of model inference with secure upload handling, account management, result persistence, and gallery-based access. The project therefore serves as a deployable prototype and a foundation for future research-oriented extensions such as rigorous quantitative evaluation, stronger classical baselines, and actual hybrid quantum-classical components.

## Introduction
Medical imaging is central to diagnosis, treatment planning, and clinical monitoring. However, medical images are frequently limited by resolution, acquisition constraints, motion, and noise. Reduced resolution can obscure fine anatomical details and lower visual clarity. Super-resolution methods aim to reconstruct higher-quality images from lower-resolution inputs while preserving important structure.

Generative adversarial networks have been widely used in image synthesis, restoration, and enhancement. In medical imaging, GAN-based approaches have shown promise in super-resolution, augmentation, restoration, segmentation support, and modality translation. In parallel, recent work in quantum and quantum-inspired generative modeling has explored whether quantum representations or hybrid quantum-classical designs can improve generative learning. This project is motivated by that research direction, but its current implementation is best described as QGAN-inspired rather than quantum-native.

## Problem Statement
Many research papers focus on model performance in isolation, while practical deployment requires a complete application pipeline. A usable system must support input validation, user access control, storage, and organized retrieval of outputs. In addition, medical-image applications benefit from domain-specific upload filtering, because irrelevant natural images should not be passed through a medical enhancement system.

This project addresses the following practical problem: how to combine a medical image enhancement pipeline with a simple but functional web application that supports secure user workflows, upload validation, result storage, and gallery management.

## Literature Survey
Recent literature shows that medical image generation and restoration are active research areas across both classical GANs and quantum-inspired methods.

Khatun et al., in *Quantum generative learning for high-resolution medical image generation* (Machine Learning: Science and Technology, 2025), proposed the Quantum Image Generative Learning framework for medical image generation. Their method uses a variational quantum circuit-based generator and a classical discriminator, together with principal-component-based compression and Wasserstein-style training ideas. They evaluated the framework on knee osteoarthritis X-rays and Medical MNIST and reported improved Fréchet Inception Distance compared with classical and prior QGAN baselines. This paper is highly relevant because it provides the main quantum-inspired motivation behind the present project, especially the idea that hybrid or quantum-oriented generative learning may improve medical image generation quality and diversity. However, the current repository does not implement variational quantum circuits, PCA-based encoding, or Wasserstein training.

The QTML 2024 conference abstract, *Quantum Generative Learning for High-Resolution Medical Image Generation*, presents an earlier version of the same research direction. It emphasizes privacy-preserving synthetic medical image generation, scalability issues in existing QGANs, and improved image quality through a quantum generator with a classical discriminator. Its importance in this project lies in supporting the research motivation rather than serving as a direct implementation blueprint.

The Springer paper *Quantum generative adversarial network for image generation* (2025) studies hybrid quantum-classical GAN design for image generation using datasets such as handwritten digits and Fashion-MNIST. The work focuses on generator and discriminator configuration, image quality, and training behavior in hybrid-QGAN settings. Although it is not medical-image-specific and does not address super-resolution deployment, it is useful as a background reference for how hybrid QGANs are structured and evaluated.

The paper *HybridQ: Hybrid Classical-Quantum Generative Adversarial Network for Skin Disease Image Generation* and its related arXiv line of work on *MediQ-GAN: Quantum-Inspired GAN for High Resolution Medical Image Generation* extend the field toward medical image generation with quantum-inspired or hybrid components. These works claim improved generation quality through mixed classical and quantum-inspired branches, latent fusion strategies, and stronger augmentation capability for medical datasets. They are relevant because they represent the frontier direction that the present project could evolve toward. At the same time, the current repository remains more modest: it uses a classical generator and applies enhancement to uploaded images rather than performing hybrid quantum image generation.

Abdulqader and Abdulazeez, in *A Comparative Study of Generative Adversarial Networks in Medical Image Processing* (Eng, 2025), provide a broader classical-GAN perspective by comparing Pix2Pix, SPADE GAN, and WGAN across multiple medical-imaging tasks. Their study uses metrics such as FID, PSNR, SSIM, Dice, and segmentation accuracy, and concludes that different architectures are suitable for different objectives. This paper is especially important for the present project because it highlights that rigorous quantitative evaluation is essential. In the current repository, such formal evaluation metrics are not yet integrated into the training or inference workflow.

Recent non-quantum medical super-resolution studies also indicate where state-of-the-art work is moving. For example, *Super-resolution of 3D medical images by generative adversarial networks with long and short-term memory and attention* (Scientific Reports, 2025) introduces a more advanced architecture for medical super-resolution that incorporates attention and sequential feature modeling. Similarly, *Transform Domain Based GAN with Deep Multi-Scale Features Fusion for Medical Image Super-Resolution* (Electronics, 2025) uses transform-domain and multi-scale feature fusion to improve reconstruction fidelity. Compared with these studies, the current repository uses a simpler residual CNN design and does not yet include attention, 3D modeling, or multi-scale fusion.

Overall, the literature suggests three major directions. First, classical GANs remain strong baselines for medical image enhancement and reconstruction. Second, quantum-inspired and hybrid quantum-classical generative models are emerging as promising research directions, particularly for data generation and augmentation. Third, recent research relies heavily on structured quantitative evaluation. The present project aligns most strongly with the practical deployment gap in the literature: it offers an end-to-end usable web prototype, while leaving quantum implementation details and formal benchmarking as future work.

## Research Gap
Based on the verified literature and the current repository, the following gap is clear:

1. Many recent papers propose advanced medical image generation or restoration models, but they do not provide a practical end-to-end user-facing system.
2. Quantum-inspired medical image generation is promising, but true hybrid implementations remain complex and are still relatively early in maturity.
3. Classical medical image super-resolution research emphasizes strong benchmarking, whereas lightweight application prototypes often omit evaluation pipelines.
4. Few prototype systems combine upload validation, user accounts, result persistence, and gallery management with a medical enhancement workflow.

## Objectives
The objectives of the present project are:

1. To build a working medical image enhancement web application.
2. To implement a QGAN-inspired super-resolution workflow using a practical classical model.
3. To validate uploaded images so that clearly non-medical images can be rejected.
4. To support user registration, login, profile management, result storage, and gallery browsing.
5. To create a foundation for future extensions such as formal benchmarking, stronger super-resolution baselines, and quantum-inspired model upgrades.

## Methodology

### 1. Web Application Layer
The application is built using FastAPI for routing and backend logic, Jinja templates for the user interface, and static assets for styling and interactivity. User sessions support login, logout, profile access, and administrative views. Uploaded and generated images are stored in local directories and made available through the web interface.

### 2. Database and Persistence
The system uses SQLAlchemy with SQLite to store user and result metadata. Each processed image record includes filename, original image path, generated result path, description, tags, patient identifier, ownership, and timestamp.

### 3. Medical Image Validation
Before enhancement, uploaded images are checked using a validation pipeline implemented in the repository. When a trained binary classifier checkpoint is available, the system uses it to distinguish medical from non-medical images. When such a checkpoint is not available, the code falls back to ImageNet-based label heuristics and color-profile heuristics to reject obvious non-medical content.

### 4. Super-Resolution Model
The repository defines a QGAN-inspired generator consisting of convolutional feature extraction, residual blocks, and progressive upsampling with pixel shuffle. A discriminator network is also defined for training support. During inference, the application attempts to load a trained generator checkpoint. If the checkpoint is missing, the application falls back to bicubic interpolation so that enhancement can still proceed.

### 5. Training Workflow
The repository includes a training script for the generator-discriminator model and a separate training script for the medical-image classifier. The super-resolution training process uses bicubic downsampling to form low-resolution inputs from high-resolution images and trains the model with adversarial and content losses. In the current codebase, the training setup is classical rather than quantum or Wasserstein-based.

## Current Implementation Status
The following claims are supported directly by the current repository:

1. The project includes a FastAPI web application for upload and result handling.
2. The project supports registration, login, logout, profile updates, password reset, gallery access, and an admin page.
3. The project stores user and result metadata in SQLite through SQLAlchemy models.
4. The project contains a medical-image classifier module and a classifier training script.
5. The project contains a QGAN-inspired generator, discriminator, and training script.
6. The project performs enhancement on uploaded images and saves both originals and processed outputs.
7. The project falls back to bicubic resizing when a trained generator checkpoint is unavailable.

The following items are discussed in the literature and may be part of future work, but are not fully implemented in the current repository:

1. true hybrid quantum-classical computation,
2. variational quantum circuit layers,
3. PCA-based feature compression for quantum generation,
4. Wasserstein or WGAN-GP style training,
5. formal evaluation using PSNR, SSIM, FID, Dice, or MOS,
6. and head-to-head comparison with stronger medical-image super-resolution baselines.

## Expected Outcome
The project is expected to provide:

1. a functional prototype for upload, validation, enhancement, and gallery management,
2. a deployable demonstration of model-assisted medical image enhancement,
3. a medical-image filtering workflow that reduces irrelevant inputs,
4. and a base system that can be extended toward stronger research evaluation and hybrid quantum-inspired experiments.

## Future Scope
Future work can extend this project in the following directions:

1. integrate quantitative evaluation metrics such as PSNR, SSIM, and FID,
2. benchmark against stronger classical baselines such as SRGAN, WGAN-based restoration, or newer medical super-resolution models,
3. explore hybrid quantum-classical model components using frameworks such as PennyLane or Qiskit,
4. add attention, multi-scale fusion, or modality-specific architectures,
5. support larger datasets and additional medical modalities,
6. and strengthen security, privacy, and production readiness for clinical-style environments.

## Conclusion
This project should be understood as a QGAN-inspired medical image enhancement prototype rather than a full quantum medical image generation system. Its present strength lies in combining a practical super-resolution pipeline with user authentication, validation, persistence, and gallery workflows in one application. The literature shows that the field is advancing toward stronger quantitative benchmarking and more sophisticated hybrid quantum-classical generative models. The current repository provides a realistic software foundation that can support that future progression.

## References
1. Khatun, A., Aydeniz, K. Y., Weinstein, Y. S., and Usman, M. *Quantum generative learning for high-resolution medical image generation*. Machine Learning: Science and Technology, 2025. https://iopscience.iop.org/article/10.1088/2632-2153/add1a9
2. Khatun, A., Aydeniz, K. Y., Weinstein, Y. S., and Usman, M. *Quantum Generative Learning for High-Resolution Medical Image Generation*. QTML 2024 abstract. https://indico.qtml2024.org/event/1/contributions/190/attachments/190/195/QTML_abstract_oral.pdf
3. *Quantum generative adversarial network for image generation*. The Visual Computer, 2025. https://link.springer.com/article/10.1007/s00371-025-03915-8
4. Jiao, Q., Tang, Y., Zhuang, J., Cong, J., and Shi, Y. *MediQ-GAN: Quantum-Inspired GAN for High Resolution Medical Image Generation*. arXiv:2506.21015, 2025. https://arxiv.org/abs/2506.21015
5. *HybridQ: Hybrid Classical-Quantum Generative Adversarial Network for Skin Disease Image Generation*. ResearchGate entry provided by user. https://www.researchgate.net/publication/393065857_HybridQ_Hybrid_Classical-Quantum_Generative_Adversarial_Network_for_Skin_Disease_Image_Generation
6. Abdulqader, M. M., and Abdulazeez, A. M. *A Comparative Study of Generative Adversarial Networks in Medical Image Processing*. Eng, 2025. https://www.mdpi.com/2673-4117/6/11/291
7. Zhang, Q. et al. *Super-resolution of 3D medical images by generative adversarial networks with long and short-term memory and attention*. Scientific Reports, 2025. https://pubmed.ncbi.nlm.nih.gov/40596207/
8. *Transform Domain Based GAN with Deep Multi-Scale Features Fusion for Medical Image Super-Resolution*. Electronics, 2025. https://www.mdpi.com/2079-9292/14/18/3726
9. *Hybrid Quantum-Classical GANs with Transfer Learning*. arXiv:2507.09706, 2025. https://arxiv.org/abs/2507.09706
