import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import time

# Image dimensions
channels = 1
latent_dim = 100
img_size = 256
img_shape = (channels, img_size, img_size)

# Create directory for generated images
output_dir = './generated_images'
os.makedirs(output_dir, exist_ok=True)

# Dataset class
class OASISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L').resize((img_size, img_size))

        if self.transform:
            image = self.transform(image)
        return image

# Generator model
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

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

def load_dataset():
    return OASISDataset(root_dir='./OASIS/keras_png_slices_train',
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])
    )

def train_gan(generator, discriminator, dataloader, n_epochs, batch_size, latent_dim, output_dir):
    adversarial_loss = nn.BCELoss().cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    Tensor = torch.cuda.FloatTensor

    for epoch in range(n_epochs):
        for i, imgs in enumerate(dataloader):
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            real_imgs = imgs.type(Tensor)

            # Train Generator
            optimizer_G.zero_grad()
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))  # Random noise
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        if (epoch + 1) % 10 == 0:
            save_images(generator, epoch, latent_dim, output_dir, Tensor)
        print(f"[Epoch: {epoch+1}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

def save_images(generator, epoch, latent_dim, output_dir, Tensor):
    sample_z = Tensor(np.random.normal(0, 1, (5, latent_dim)))  # Generate 5 images
    sample_gen_imgs = generator(sample_z).detach().cpu()

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for idx in range(5):
        axes[idx].imshow(sample_gen_imgs[idx][0], cmap='gray')
        axes[idx].axis('off')
    plt.savefig(os.path.join(output_dir, f'generated_brains_epoch_{epoch+1}.png'), bbox_inches='tight')
    plt.close(fig)

    torch.save(generator.state_dict(), os.path.join(output_dir, f'generator_epoch_{epoch+1}.pth'))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, f'discriminator_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    start_time = time.time()

    dataset = load_dataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()

    train_gan(generator, discriminator, dataloader, n_epochs=5, batch_size=64, latent_dim=100, output_dir=output_dir)

    elapsed_time = time.time() - start_time
    elapsed_hours = elapsed_time // 3600
    elapsed_minutes = (elapsed_time % 3600) // 60
    elapsed_seconds = elapsed_time % 60

    print("Training complete. Models and generated images are saved.")
    print(f"Training time: {int(elapsed_hours)} hours, {int(elapsed_minutes)} minutes, and {int(elapsed_seconds)} seconds")

