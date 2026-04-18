# Viva Question Bank

## Project Positioning

### 1. Why do you call the project QGAN-inspired instead of a true QGAN?
The project is called QGAN-inspired because its research motivation comes from recent quantum generative learning papers, especially work on hybrid quantum-classical image generation. However, the current repository implements a practical classical convolutional generator rather than variational quantum circuits or quantum hardware.

### 2. What is the biggest difference between your project and the latest quantum medical image papers?
The latest papers focus on novel hybrid or quantum-inspired model design and quantitative benchmarking, while my project focuses on building an end-to-end usable prototype with upload, validation, authentication, storage, and gallery workflows.

### 3. What is the strongest honest claim you can make about your project?
It is an end-to-end medical image enhancement web application with a QGAN-inspired research foundation and a classical implementation.

## Literature-Based Questions

### 4. What is QIGL and why is it important to your project?
QIGL stands for Quantum Image Generative Learning. It is important because it provides the main research direction behind the project’s QGAN-inspired framing. It uses a quantum generator and classical discriminator for medical image generation.

### 5. Did you implement the same QIGL method in your repository?
No. My repository does not implement the full QIGL method. It does not include variational quantum circuits, PCA-based encoding, or Wasserstein-based training. It uses a classical generator inspired by that direction.

### 6. What role does the QTML 2024 abstract play in your literature review?
It supports the same research direction as the later QIGL paper and shows that the topic of quantum generative learning for medical images is academically active and recent.

### 7. How is the Springer QGAN paper relevant if it is not medical?
It is useful for conceptual background. It helps explain how hybrid quantum-classical GANs are structured, even though it is not directly a medical super-resolution system.

### 8. What do HybridQ and MediQ-GAN contribute to the field?
They push the field toward hybrid or quantum-inspired medical image generation with more advanced architectures and stronger claims about image quality and augmentation capability.

### 9. What does the MDPI comparative GAN study teach you?
It shows that no single GAN architecture is universally best. Model selection must depend on the task, and formal evaluation with PSNR, SSIM, FID, and related metrics is essential.

### 10. Why are the newer medical super-resolution papers important for your project?
They show where the field is moving: toward attention mechanisms, multi-scale fusion, stronger evaluation, and more advanced reconstruction models.

## Model and Implementation Questions

### 11. What model do you actually use in your code?
I use a classical convolutional generator with residual blocks and pixel-shuffle upsampling, along with a discriminator definition for adversarial training. In the current repository, the enhancement pipeline is configured for 8x super-resolution.

### 12. What is the function of the discriminator in your project?
The discriminator is used in the training pipeline to distinguish real high-resolution images from generated ones, helping the generator learn more realistic outputs.

### 13. What happens if the trained generator file is missing?
The application falls back to bicubic interpolation so the enhancement pipeline can still function.

### 14. Why did you choose a classical model instead of a real quantum implementation?
Because the goal of the current stage was to build a usable and working prototype. A classical implementation is easier to train, deploy, and integrate into a web application.

### 15. What is the role of pixel shuffle in your generator?
Pixel shuffle is used for upsampling. It rearranges feature channels into higher-resolution spatial output and is commonly used in super-resolution models.

## Medical Validation Questions

### 16. Why do you validate uploads before enhancement?
Because the platform is intended for medical scan images. Validation helps reject irrelevant natural images and makes the workflow more domain-specific.

### 17. How does your medical image validation work?
It uses a custom binary classifier if trained weights are available. If not, it falls back to ImageNet-based label heuristics and simple image-profile heuristics such as grayscale and saturation checks.

### 18. Is this validation medically reliable enough for clinical use?
No. It is a prototype-level filtering mechanism, not a clinically validated diagnostic tool.

## Evaluation Questions

### 19. Which evaluation metrics are common in the literature you reviewed?
The most common metrics are PSNR, SSIM, FID, Dice coefficient, segmentation accuracy, and sometimes qualitative or expert-based review.

### 20. Does your current repository compute these metrics?
Not yet in a formal integrated way. That is one of the main future improvements needed to align the project more closely with the literature.

### 21. Why is benchmarking important?
Because without benchmarking, it is difficult to compare the model against bicubic interpolation or stronger GAN baselines and to justify claims about image quality.

## System Questions

### 22. Why did you build a web application instead of only training a model?
Because many research projects stop at training results. I wanted to show how a medical image enhancement model can be integrated into a usable system with upload, storage, authentication, and gallery management.

### 23. What technologies are used in your project?
Python, FastAPI, SQLAlchemy, SQLite, PyTorch, Jinja templates, HTML, CSS, and JavaScript.

### 24. Does your current project support DICOM or NIfTI files?
No. The current implementation supports PNG, JPG, and JPEG uploads. DICOM and NIfTI can be considered future extensions.

### 25. What does the gallery add to the project?
It makes the system practical by allowing users to revisit, search, download, and manage their processed results.

### 26. Why is your project still useful even though it is not a true quantum system?
Because it bridges research motivation and practical deployment. It shows how a medically oriented enhancement workflow can be built today while leaving room for future quantum-inspired upgrades.

## Limitation and Future Scope Questions

### 27. What are the main limitations of your current work?
The main limitations are the lack of true hybrid quantum implementation, the absence of formal benchmark metrics in the current workflow, the lack of DICOM and NIfTI support, and the fact that the model is simpler than recent state-of-the-art medical SR architectures.

### 28. What is your most important future improvement?
Adding rigorous evaluation with PSNR, SSIM, and FID, along with comparison against stronger classical baselines.

### 29. How could you make the project more aligned with the QGAN literature?
By experimenting with hybrid quantum-classical layers, variational quantum circuits, or quantum-inspired latent-space modules using frameworks such as PennyLane or Qiskit.

### 30. How could you improve the super-resolution model itself?
By adding attention, multi-scale fusion, modality-specific design, or stronger training strategies such as better losses and benchmarked baselines.

### 31. What is the main academic contribution of your current project?
Its main contribution is a practical prototype that integrates medical-image validation, super-resolution inference, user workflows, and result management in one system.
