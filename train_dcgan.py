import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dcgan import Generator, Discriminator

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataloader = torch.utils.data.DataLoader(
    datasets.CelebA(root='./data', split='train', download=True, transform=transform),
    batch_size=128, shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

for epoch in range(10):
    for i, (real, _) in enumerate(dataloader):
        b_size = real.size(0)
        real = real.to(device)

        label_real = torch.ones(b_size, device=device)
        label_fake = torch.zeros(b_size, device=device)

        netD.zero_grad()
        output = netD(real)
        errD_real = criterion(output, label_real)

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        errD_fake = criterion(output, label_fake)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        output = netD(fake)
        errG = criterion(output, label_real)
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"Epoch {epoch}, Step {i}, D_loss: {errD.item()}, G_loss: {errG.item()}")
            save_image(fake[:25], f"results/fake_{epoch}_{i}.png", nrow=5, normalize=True)
