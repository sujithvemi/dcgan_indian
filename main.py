import argparse
import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms, datasets
# import torchvision.utils as vutils
from model import Discriminator, Generator
from utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='lfw | lfw_indian')
parser.add_argument('--dataroot', required=True, help='location of data')
parser.add_argument('--nc', required=True, help='number of channels in the image (depends on color or grayscale)')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--nz', type=int, default=100, help='the length of the latent vector used as input to generator')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--gckpt', default='', help='path to the state dict of generator module (to resume training if available)')
parser.add_argument('--dckpt', default='', help='path to the state dict of discriminator module (to resume training if available)')
# parser.add_argument('--outDir', default='outputs', help='folder to save output images and model checkpoints (in separate subfolders)')
parser.add_argument('--seed', type=int, help='random seed for reproducibility')
parser.add_argument('--numSamples', type=int, default=16, help='number of test samples to be generated for checking the model training visually')
parser.add_argument('--startEpoch', type=int, help='the epoch from where training should resume')

opt = parser.parse_args()

# if not os.path.isdir(outDir):
#     os.makedirs(outDir + '/images/indian')
#     os.makedirs(outDir + '/images/default')
#     os.makedirs(outDir + '/models/indian')
#     os.makedirs(outDir + '/models/default')

if not opt.seed:
    opt.seed = random.randint(1, 1000)
seed = int(opt.seed)
print('random seed selected: ', seed)
random.seed(seed)
torch.manual_seed(seed)

cudnn.benchmark = True

# if dataset == 'lfw_indian':
#     images_dest = outDir + '/images/indian/'
#     models_dest = outDir + '/models/indian/'
# else:
#     images_dest = outDir + '/images/default/'
#     models_dest = outDir + '/models/default/'
dataset = datasets.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(int(opt.imageSize)),
                                   transforms.CenterCrop(int(opt.imageSize)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
nc = int(opt.nc)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(opt.batchSize), shuffle=True)
num_batches = len(dataloader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
lr = float(opt.lr)
beta1 = float(opt.beta1)
num_epochs = int(opt.epochs)
if opt.startEpoch is not None:
    start_epoch = int(opt.startEpoch)
else:
    start_epoch = 0

def weights_init(layer):
    layer_class = layer.__class__.__name__
    if layer_class.lower().find('conv') != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_class.lower().find('batchnorm') != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

generator = Generator(nc, nz, ngf).to(device)
generator.apply(weights_init)
if opt.gckpt != '':
    generator.load_state_dict(torch.load(opt.gckpt))
print('Generator')
print(generator)

discriminator = Discriminator(nc, ndf).to(device)
discriminator.apply(weights_init)
if opt.dckpt != '':
    discriminator.load_state_dict(torch.load(opt.dckpt))
print('Discriminator')
print(discriminator)
print('\n')
def noise(samples):
    n = torch.autograd.Variable(torch.randn(samples, nz, 1, 1))
    if torch.cuda.is_available(): return n.cuda()
    return n
test_noise = noise(int(opt.numSamples))
real_label = 1
fake_label = 0

loss = nn.BCELoss()

d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

logger = Logger(model_name='DCGAN', data_name=opt.dataset)
print('training starts')
pbar = tqdm(total=num_batches*num_epochs)
for epoch in range(start_epoch, start_epoch+num_epochs):
    for n_batch, (real_batch,_) in enumerate(dataloader):
        d_optim.zero_grad()
        real_data = real_batch.to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        d_real_out = discriminator(real_data)
        d_real_err = loss(d_real_out, label)
        d_real_err.backward()
        d_x = d_real_out.mean().item()

        fake_data = generator(torch.randn(batch_size, nz, 1, 1, device=device))
        label.fill_(fake_label)
        d_fake_out = discriminator(fake_data.detach())
        d_fake_err = loss(d_fake_out, label)
        d_fake_err.backward()
        D_G_z1 = d_fake_out.mean().item()

        d_tot_err = d_real_err + d_fake_err
        d_optim.step()

        g_optim.zero_grad()
        label.fill_(real_label)
        g_out = discriminator(fake_data)
        g_err = loss(g_out, label)
        g_err.backward()
        D_G_z2 = g_out.mean().item()
        g_optim.step()

        logger.log(d_tot_err, g_err, epoch, n_batch, num_batches)
        
        if n_batch % 100 == 0:
            test_images = generator(test_noise).data.cpu()
            logger.log_images(test_images, int(opt.numSamples), epoch, n_batch, num_batches)
            logger.display_status(epoch, num_epochs, n_batch, num_batches, d_tot_err, g_err, d_real_out, d_fake_out)
        pbar.update(1)
        logger.save_models(generator, discriminator, epoch)
pbar.close()