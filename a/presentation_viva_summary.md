# Presentation and Viva Summary

## One-Minute Summary
My project is a medical image super-resolution web application. It allows users to upload medical scans, validates whether the uploaded image is medical in nature, enhances the image using a QGAN-inspired super-resolution model, and stores the results in a searchable gallery. The system is built with FastAPI, SQLAlchemy, SQLite, and a template-based frontend. The main goal is to combine AI-based image enhancement with a usable and secure end-to-end application.

## Problem the Project Solves
Medical images may suffer from low resolution, blur, and loss of detail. This can reduce visual clarity and make analysis more difficult. Many research models focus only on training performance, but they do not provide a complete working system with upload, filtering, storage, and result management. My project addresses this by combining model inference with a full web application workflow.

## Main Objectives
1. To enhance low-resolution medical images.
2. To reject non-medical image uploads.
3. To provide secure user login and gallery management.
4. To store and organize enhanced results for later access.

## Technologies Used
1. Python
2. FastAPI
3. SQLAlchemy
4. SQLite
5. PyTorch
6. Jinja2 templates
7. HTML, CSS, and JavaScript

## Core Technology
Three main pillars define the current implementation.

### 1. QGAN-Inspired Classical Architecture
The project uses a PyTorch-based generator and discriminator for adversarial super-resolution training. The current implementation is inspired by QGAN research, but the deployed code is a classical convolutional design rather than a true hybrid quantum-classical model.

### 2. Medical Image-Oriented Workflow
The system is designed for medical-image upload and enhancement. It includes a medical-image validation step before processing and currently accepts PNG, JPG, and JPEG files. The workflow is aimed at scans such as MRI, CT, and X-ray images, but DICOM and NIfTI support are not yet implemented in the present codebase.

### 3. 8x Super-Resolution
The application enhances images using an 8x super-resolution pipeline. A generator network produces the enhanced output, and a discriminator is used during training to improve realism through adversarial learning.

## System Workflow
1. The user registers or logs in.
2. The user uploads an image through the web interface.
3. The system checks the file type and validates whether it is a medical image.
4. If valid, the image is passed to the QGAN-inspired generator.
5. The enhanced image is saved in the results folder.
6. Metadata is stored in the database.
7. The user can later view, search, download, or manage results in the gallery.

## Model Explanation
The super-resolution model is called QGAN-inspired because the project is based on ideas from quantum generative learning literature. In the current implementation, the actual model is a classical deep learning architecture built using convolution layers, residual blocks, and pixel-shuffle upsampling. A discriminator is also defined for adversarial training. The deployed system currently performs 8x enhancement and uses a practical classical implementation rather than a true quantum model.

## Medical Image Validation
The project does not directly accept every uploaded image. It includes a validation step that checks whether the image is likely to be a medical scan. This is done using:

1. a custom binary classifier if trained weights are available,
2. fallback ImageNet-based label checking,
3. and color and grayscale heuristics.

This helps reduce irrelevant uploads and makes the pipeline more domain-specific.

## Key Features
1. User registration and login
2. Session-based authentication
3. Password reset flow
4. 8x image upload and enhancement
5. Original-versus-enhanced result display
6. Searchable gallery
7. Admin dashboard
8. Download support for results

## What Makes the Project Different
1. It combines AI model inference with a working web platform.
2. It includes medical-image validation before enhancement.
3. It uses adversarial training with a generator and discriminator for 8x super-resolution.
4. It leaves room for future hybrid quantum-classical extension.

## Current Limitations
1. The current implementation is quantum-inspired, not a true quantum-computing model.
2. Formal metrics such as PSNR, SSIM, and FID are not yet fully integrated into the app workflow.
3. The quality of enhancement depends on the available training data and checkpoint quality.
4. The project is a prototype and still needs stronger production-level security hardening.

## Future Scope
1. Add PSNR, SSIM, and FID evaluation.
2. Compare against more classical baselines.
3. Integrate PennyLane or Qiskit for actual hybrid quantum experimentation.
4. Improve security and compliance for clinical deployment.
5. Support more medical modalities and larger datasets.

## Likely Viva Questions and Short Answers

### 1. Why did you choose this topic?
I chose this topic because medical images often suffer from low resolution, and improving visual quality can support better inspection. I also wanted to combine AI research with a practical web application instead of building only a standalone model.

### 2. Why do you call it QGAN-inspired?
I call it QGAN-inspired because the project direction and literature motivation come from quantum generative learning research. However, the current implementation is classical and uses convolutional neural networks for practical deployment.

### 3. What is the role of the classifier?
The classifier checks whether the uploaded image looks like a medical scan. This prevents irrelevant images from entering the enhancement pipeline.

### 4. What is the role of FastAPI in the project?
FastAPI handles routing, upload processing, authentication-related endpoints, and API responses. It acts as the backend layer connecting the model, database, and frontend.

### 5. Why did you use SQLite?
SQLite is lightweight, simple to set up, and suitable for a prototype or academic demonstration project.

### 6. What happens if the trained model is missing?
If the trained generator checkpoint is not available, the application falls back to bicubic interpolation so the image can still be upscaled.

### 7. How is this project useful?
It demonstrates how AI-based medical image enhancement can be turned into a usable system with validation, storage, and user access features.

### 8. What are the major limitations?
The main limitations are the absence of formal benchmark evaluation in the current code, the lack of true quantum implementation, the absence of DICOM and NIfTI support, and the need for stronger production security.

## Closing Statement
In summary, this project is an end-to-end prototype for medical image enhancement using a QGAN-inspired super-resolution approach. It combines machine learning, validation, secure access, and gallery management in one application, and it provides a practical base for future academic and technical improvements.

## Supporting Files
For final report and presentation preparation, see:

1. [literature_survey_final.md](/home/pinkee-sharma/qgan-superres-website/literature_survey_final.md)
2. [paper_vs_project_comparison.md](/home/pinkee-sharma/qgan-superres-website/paper_vs_project_comparison.md)
3. [viva_question_bank.md](/home/pinkee-sharma/qgan-superres-website/viva_question_bank.md)
4. [ppt_outline_5_slides.md](/home/pinkee-sharma/qgan-superres-website/ppt_outline_5_slides.md)
