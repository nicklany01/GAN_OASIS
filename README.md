# GAN Brain Image Generator

This project implements a **Generative Adversarial Network (GAN)** using PyTorch to generate realistic brain images from the OASIS dataset. The model consists of two networks: a **Generator** and a **Discriminator**, trained adversarially to create images that resemble real brain images.

## Project Overview

The GAN aims to generate synthetic brain images based on the preprocessed OASIS dataset, which contains grayscale brain scan images. The **Generator** generates images from random noise, while the **Discriminator** attempts to classify images as real (from the dataset) or fake (generated).

## How GANs Work

A **Generative Adversarial Network (GAN)** consists of two networks:
- **Generator**: Takes a random vector (latent space) and generates images that resemble real data.
- **Discriminator**: Classifies images as either real or fake. It attempts to differentiate between genuine images from the dataset and those produced by the generator.

The two networks are trained together in a zero-sum game: the generator improves at fooling the discriminator, and the discriminator improves at identifying fake images.

## Dataset

The project uses the **OASIS Brain MRI Dataset**, specifically the preprocessed slices available in the `./OASIS/keras_png_slices_train/` directory. Ensure that the images are grayscale (single channel) and in `.png` format.

## Dependencies

Ensure you have the following dependencies installed:
- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Torchvision
- PIL (Python Imaging Library)

You can install the dependencies via:

```bash
pip install torch torchvision numpy matplotlib pillow
