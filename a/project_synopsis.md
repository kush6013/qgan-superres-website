# Project Synopsis

## Project Title
QGAN-Based Medical Image Super-Resolution and Secure Gallery System

## Broad Area of Work
Medical Image Processing and Artificial Intelligence

This project belongs to the broad area of medical image processing, deep learning, and quantum-inspired generative modeling. It combines:
- medical image enhancement and super-resolution,
- generative adversarial networks (GANs),
- quantum-inspired or hybrid quantum-classical generative architectures,
- secure web application development for upload, processing, and gallery management.

The work spans research in AI for healthcare, data augmentation, and image quality improvement with model architectures that are suitable for emerging quantum machine learning trends.

## Introduction
Medical imaging plays a critical role in diagnosis, treatment planning, and monitoring of diseases. Common imaging modalities such as X-ray, MRI, CT scan, and ultrasound produce images that may suffer from low resolution, noise, and loss of fine detail. Super-resolution techniques aim to improve the spatial resolution of images while preserving clinically relevant features.

Generative adversarial networks (GANs) have demonstrated exceptional ability in generating and enhancing realistic images. Medical GAN models can generate high-quality synthetic scans and improve image resolution for better diagnostic use. However, existing GAN-based medical methods often struggle with stability, high-resolution scalability, and the scarcity of annotated medical data.

Recent research has proposed quantum or hybrid quantum-classical generative approaches to address some of these limitations. Quantum image generative learning can offer more expressive feature extraction and potentially improved training dynamics. This project builds on these ideas by implementing a QGAN-inspired model for medical image super-resolution, coupled with a web-based system for secure upload, validation, enhancement, and gallery management.

## Literature Survey / Work Done in the Field
Recent research in medical image generation and enhancement shows two major trends. The first is the continued use of classical generative adversarial networks for restoration, segmentation support, and super-resolution. The second is the emergence of quantum-inspired and hybrid quantum-classical generative models that aim to improve generative learning efficiency, image diversity, and representational capability. The present project is positioned between these two directions: it is inspired by recent QGAN literature, but its current implementation is a practical classical prototype deployed through a web application.

One of the most relevant studies is the work of Khatun et al., *Quantum generative learning for high-resolution medical image generation* (2025). This paper proposes the Quantum Image Generative Learning (QIGL) framework, which combines a variational quantum circuit-based generator with a classical discriminator for medical image generation. The study addresses limitations in earlier QGANs, particularly poor scalability, patch-based generation, and unstable training. Instead of relying only on local pixel patches, the method uses principal-component-based image representation and incorporates Wasserstein-style training ideas to improve diversity and stability. The model was evaluated on knee osteoarthritis X-rays and the Medical MNIST dataset, with Fréchet Inception Distance used to compare performance against classical and prior quantum baselines. This paper is highly important for the present project because it provides the main research basis for describing the work as QGAN-inspired. However, the current repository does not yet implement variational quantum circuits, PCA-based encoding, or FID-based benchmarking.

The QTML 2024 abstract titled *Quantum Generative Learning for High-Resolution Medical Image Generation* presents an earlier conference version of the same research direction. It emphasizes the importance of synthetic medical image generation in privacy-sensitive domains and highlights the challenges faced by existing QGANs, including poor scalability and low image quality. The abstract reinforces the value of a quantum generator combined with a classical discriminator and supports the broader claim that quantum-inspired generative methods are becoming relevant in medical imaging research. For the present project, this abstract is useful as supporting literature for the project’s motivation, even though it does not directly map to the deployed code.

Another useful quantum-related reference is the Springer paper *Quantum generative adversarial network for image generation* (2025). This study focuses on hybrid quantum-classical QGAN design and evaluates image generation quality on datasets such as handwritten digits and Fashion-MNIST. The paper is not specific to medical imaging and does not focus on super-resolution, but it is still relevant because it explains how hybrid quantum generators and classical discriminators are structured and compared. It therefore supports the conceptual background of the project, particularly when explaining QGAN terminology in the synopsis or viva.

The direction of hybrid or quantum-inspired medical image generation is further reflected in *HybridQ: Hybrid Classical-Quantum Generative Adversarial Network for Skin Disease Image Generation* and the related arXiv paper *MediQ-GAN: Quantum-Inspired GAN for High Resolution Medical Image Generation* (2025). These works push beyond general QGAN design toward medical-image-specific generation, data augmentation, and higher-quality synthesis. MediQ-GAN, in particular, describes a dual-stream quantum-inspired and classical generator architecture, prototype-guided skip connections, and latent-space design choices aimed at improving image quality and trainability. These papers are relevant to the present work as forward-looking references. They show the type of hybrid architecture that a future version of this project could adopt. At the same time, they also make it clear that the current repository should be described honestly as quantum-inspired in motivation, not quantum-implemented in practice.

From the classical medical GAN perspective, the paper by Abdulqader and Abdulazeez, *A Comparative Study of Generative Adversarial Networks in Medical Image Processing* (2025), is especially valuable. This study compares Pix2Pix, SPADE GAN, and WGAN across cardiac MRI, brain tumor MRI, and abdominal MRI tasks. It evaluates the models using FID, PSNR, SSIM, Dice coefficient, and segmentation accuracy. The main takeaway is that no single GAN architecture performs best for every medical imaging task; the correct choice depends on whether the goal is synthesis, restoration, inpainting, segmentation, or enhancement. This finding is directly useful for the present project because it shows that model evaluation should be aligned with the task. It also highlights a current limitation of the repository: although the application performs enhancement and gallery management, it does not yet include a formal quantitative evaluation pipeline using PSNR, SSIM, or FID.

Recent medical super-resolution research also demonstrates that the field is moving toward more advanced classical architectures. For instance, *Super-resolution of 3D medical images by generative adversarial networks with long and short-term memory and attention* (2025) introduces a more sophisticated design for medical image super-resolution by incorporating attention and sequence-aware modeling. Similarly, *Transform Domain Based GAN with Deep Multi-Scale Features Fusion for Medical Image Super-Resolution* (2025) focuses on transform-domain learning and multi-scale feature fusion to improve reconstruction fidelity. These studies are important because they show that state-of-the-art medical super-resolution now often includes attention mechanisms, multi-scale design, and structured evaluation with PSNR and SSIM. Compared with these works, the present project uses a simpler residual CNN-based generator, which makes it easier to deploy but less advanced in research novelty.

An additional relevant trend appears in more recent hybrid quantum-classical studies such as *Hybrid Quantum-Classical GANs with Transfer Learning* (2025). This line of work explores how transfer learning and hybrid placement of quantum and classical components affect image generation quality and training behavior. Such studies are useful for defining future scope, because they suggest that hybrid architectures may benefit from careful partitioning of the generator and discriminator rather than simply replacing classical layers with quantum ones.

Taken together, the verified literature suggests the following. First, classical GANs remain strong and practical baselines for medical image restoration and enhancement. Second, quantum-inspired and hybrid quantum-classical generative models are emerging as promising but still developing research directions, particularly in medical image generation and augmentation. Third, recent papers rely heavily on structured quantitative benchmarking, especially with metrics such as FID, PSNR, and SSIM. The present project contributes most clearly in a different but valuable way: it provides an end-to-end application that integrates upload validation, user management, result persistence, and gallery-based access around a medical image enhancement pipeline. Therefore, the project is best positioned as a QGAN-inspired medical image super-resolution web prototype with strong practical deployment features and clear future scope for stronger benchmarking and hybrid quantum-classical extension.

## Existing Gaps
Based on the literature and current state of medical image super-resolution, the following gaps are identified:

1. **High-resolution medical image generation with GANs is still unstable**
   - Many GAN models work for low-resolution or structured datasets, but quality degrades when scaling to clinically relevant resolutions.
2. **Limited adoption of quantum-inspired models in medical imaging**
   - Existing quantum GAN research is primarily theoretical or limited in real-world medical evaluation.
3. **Lack of practical end-to-end systems**
   - Few works combine the model with a usable web-based upload and gallery system for medical users.
4. **Insufficient medical image validation**
   - GAN pipelines often do not enforce strong medical image filtering, which can allow non-medical images or irrelevant scans to pass through.
5. **Dataset scarcity and privacy constraints**
   - Medical datasets are limited, and generating privacy-safe synthetic images remains a challenge.
6. **Few robust comparisons to classical baselines**
   - More comparative analysis is needed to prove quantum-inspired models outperform or complement classical GANs.

## Objectives of the Proposed Work
The primary objectives of this project are:

1. Develop a medical image super-resolution pipeline using QGAN-inspired techniques.
2. Build a secure web application for uploading, validating, enhancing, and managing medical image scans.
3. Implement a medical scan classifier to accept only medical images (PNG/JPG/JPEG) and reject irrelevant content.
4. Compare model output quality to classical baselines using PSNR, SSIM, and FID metrics.
5. Create a dataset preparation and training workflow suitable for medical imaging.
6. Demonstrate the system on medical image examples such as CT scans, X-rays, or dermatology images.
7. Prepare a complete project synopsis and thesis document that covers research, implementation, and results.

## Proposed Methodologies
This project will use a combination of data engineering, machine learning, and web application development methodologies.

### 1. Data Collection and Preprocessing
- Collect medical images from public datasets where available.
- Organize images into training and validation folders:
  - `data/medical/train/medical`
  - `data/medical/train/nonmedical`
  - `data/medical/val/medical`
  - `data/medical/val/nonmedical`
- Preprocess images with resizing, cropping, normalization, and augmentation.
- Prepare low-resolution input images using bicubic downsampling for super-resolution training.

### 2. Medical Image Validation
- Implement a classifier to verify the uploaded image is a medical scan.
- Use a hybrid approach with:
  - pre-trained ImageNet-based feature heuristics,
  - a custom dataset-trained binary classifier,
  - saturation and grayscale heuristics to identify scan-like content.
- Reject non-medical uploads before enhancement.

### 3. QGAN-Inspired Model Design
- Design a hybrid generator architecture inspired by the IOP and Springer papers.
- Use classical convolutional layers and optional quantum-inspired blocks in the generator.
- Use a classical discriminator to judge realism.
- Incorporate PCA-based encoding and Wasserstein loss ideas to improve scalability.

### 4. Training Strategy
- Train with a combined GAN loss:
  - adversarial loss for realism,
  - content loss (L1) for structural fidelity,
  - optional perceptual features for sharper results.
- Use batch-based training on medical images with augmentation.
- Save model weights for use in the web application.

### 5. Web Application Implementation
- Implement the app using FastAPI and SQLAlchemy.
- Features:
  - secure image upload form,
  - drag-and-drop file input,
  - medical image validation before processing,
  - QGAN super-resolution enhancement,
  - user session and gallery management,
  - delete and download options for enhanced results.
- Use templates for responsive UI and show original vs. enhanced images.

### 6. Evaluation and Comparison
- Evaluate image quality using:
  - Peak Signal-to-Noise Ratio (PSNR),
  - Structural Similarity Index (SSIM),
  - Fréchet Inception Distance (FID),
  - visual assessment of enhanced images.
- Compare the QGAN-inspired model to:
  - bicubic interpolation baseline,
  - classical GAN-based super-resolution baseline.

### 7. Synopsis Documentation
- Compile a complete project synopsis document.
- Cover introduction, review, methodology, experiments, results, and future scope.

## Expected Outcome of the Proposed Work
The following outcomes are expected:

1. A working web application for medical image uploads and QGAN enhancement.
2. A trained model capable of improving image resolution for medical scans.
3. A robust medical image validation pipeline to reject non-medical uploads.
4. Comparative performance results against classical baselines.
6. A complete project synopsis draft with detailed literature review and implementation notes.
6. Repository code and documentation for reproducibility.

## Future Scope of the Work
The project can be extended in several directions:

1. **Real Quantum Hardware**
   - Move some model operations to actual quantum processors using Qiskit or Pennylane.
2. **Multi-modal medical images**
   - Extend support to MRI, CT, ultrasound, dermatoscopic, and histopathology images.
3. **Explainability**
   - Add explainable AI features to interpret enhanced medical images.
4. **Regulatory compliance**
   - Add privacy-preserving mechanisms and HIPAA/GDPR alignment for clinical use.
5. **Mobile deployment**
   - Create a lightweight mobile interface for doctors to use the enhancement tool.
6. **Advanced evaluation**
   - Use domain expert review and clinical validation studies.
7. **Augmentation for diagnosis**
   - Use generated images to augment training data for other diagnostic AI models.

## References
1. Amena Khatun, Kübra Yeter Aydeniz, Yaakov S Weinstein, Muhammad Usman. *Quantum generative learning for high-resolution medical image generation*. Mach. Learn.: Sci. Technol. 6 (2025) 025032.
2. QTML 2024 abstract: *Quantum Generative Learning for High-Resolution Medical Image Generation*.
3. *A Comparative Study of Generative Adversarial Networks in Medical Image Processing*, MDPI.
4. *Quantum generative adversarial network for image generation*, Springer.
5. HybridQ research paper (ResearchGate) - Hybrid Classical-Quantum Generative Adversarial Network for Skin Disease Image Generation.
6. Goodfellow, I., et al. *Generative Adversarial Nets*. Advances in Neural Information Processing Systems, 2014.
7. Ledig, C., et al. *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*. CVPR 2017.
8. Wang, Z., et al. *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE TIP 2004.
9. Kingma, D. P., & Ba, J. *Adam: A Method for Stochastic Optimization*. ICLR 2015.

---

### Notes
- The references are based on the papers you provided and the relevant field of work.
- The proposed structure is ready to be expanded into a full synopsis document by adding implementation details, equations, results, tables, and figures.
- I can also create a more detailed `synopsis.docx`-style outline or a PowerPoint-ready version if needed.
