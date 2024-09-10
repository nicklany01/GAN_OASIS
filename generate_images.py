import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the Generator model (same architecture used for training)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(input_features, output_features, normalize=True):
            layers = [nn.Linear(input_features, output_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

# Configuration
latent_dim = 100
img_size = 256
img_shape = (1, img_size, img_size)  # Channels, Height, Width
output_dir = './generated_images'
num_images = 5  # Number of images to generate

# Initialize Generator
generator = Generator().cuda()
generator.load_state_dict(torch.load(os.path.join(output_dir, 'generator.pth')))
generator.eval()

# Generate images
def generate_images(generator, num_images, latent_dim, img_shape, output_dir):
    with torch.no_grad():
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (num_images, latent_dim)))  # Random noise
        gen_imgs = generator(z).cpu()

        # Save generated images
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for idx in range(num_images):
            axes[idx].imshow(gen_imgs[idx][0], cmap='gray')
            axes[idx].axis('off')
        plt.savefig(os.path.join(output_dir, 'generated_images.png'), bbox_inches='tight')
        plt.close(fig)

# Generate and save images
generate_images(generator, num_images, latent_dim, img_shape, output_dir)
print(f"Generated images saved to {os.path.join(output_dir, 'generated_images.png')}")

