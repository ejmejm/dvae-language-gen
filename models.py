import os
from PIL import Image
import random
import string

from einops import rearrange
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn import functional as F

CHAR_SET = list(string.ascii_lowercase) + [' ']

# https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999
class ImgDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        self.img_cache = [None] * len(self.all_imgs)

    def shuffle(self):
        zipped_items = list(zip(self.all_imgs, self.img_cache))
        random.shuffle(zipped_items)
        self.all_imgs, self.img_cache = zip(*zipped_items)
        self.all_imgs = list(self.all_imgs)
        self.img_cache = list(self.img_cache)
        
    def get_batch_iter(self, batch_size):
        for i in range(0, len(self.all_imgs), batch_size):
            yield self[i:i+batch_size]
        
    def __len__(self):
        return len(self.all_imgs)

    def _load_img(self, idx):
        if self.img_cache[idx] is not None:
            return self.img_cache[idx]
        
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_img = self.transform(image)
        self.img_cache[idx] = tensor_img
        
        return tensor_img
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._load_img(idx)
        elif isinstance(idx, slice):
            start = idx.start or 0
            end = idx.stop or len(self.all_imgs)
            step = idx.step or 1
            
            if start < 0:
                start = len(self.all_imgs) + start
            if end < 0:
                end = len(self.all_imgs) + end
            end = min(end, len(self.all_imgs))
            
            imgs = []
            for i in range(start, end, step):
                iter_img = self._load_img(i).unsqueeze(0)
                imgs.append(iter_img)
            imgs_tensor = torch.cat(imgs, dim=0)
            return imgs_tensor

class ResidualBlock(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, n_filters, kernel_size=3):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError('Residual blocks must use and odd kernel size!')

        self.n_filters = n_filters
        padding = int(kernel_size // 2)

        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size, 1, padding)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size, 1, padding)

        self.batch_norm1 = nn.BatchNorm2d(n_filters)
        self.batch_norm2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        # Convolution 1
        z = self.batch_norm1(x)
        z = F.relu(z)
        z = self.conv1(z)

        # Convolution 2
        z = self.batch_norm2(z)
        z = F.relu(z)
        z = self.conv2(z)
        z += x

        return z
        
class Encoder(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim
        
        input_filter_count = 2 * img_dim[0]
        
        self.downsampling_layer1 = nn.Sequential(
                nn.Conv2d(input_filter_count, 32, 4, stride=2, padding=1),
                nn.ReLU())
        self.downsampling_layer2 = nn.Sequential(
                nn.Conv2d(32, 32, 4, stride=2, padding=1),
                nn.ReLU())
        self.downsampling_layer3 = nn.Sequential(
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU())
        self.downsampling_layer4 = nn.Sequential(
                nn.Conv2d(64, 64, 4, stride=2, padding=1),
                nn.ReLU())
        self.downsampling_layer5 = nn.Sequential(
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU())
        
        self.res_block = ResidualBlock(128)
        
        self.conv_layer = nn.Conv2d(128, len(CHAR_SET), 1)
        
    def forward(self, target_img, curr_img):
        full_input = torch.cat([target_img, curr_img], dim=1)
        
        z = self.downsampling_layer1(full_input)
        z = self.downsampling_layer2(z)
        z = self.downsampling_layer3(z)
        z = self.downsampling_layer4(z)
        z = self.downsampling_layer5(z)
        z = self.res_block(z)
        logits = self.conv_layer(z)
        one_hot_tokens = F.gumbel_softmax(logits, hard=True)
        
        return one_hot_tokens
    
class Decoder(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.img_dim = img_dim
        self.embeddings_dim = 64
        assert img_dim[1] == img_dim[2], 'Only supports square imgs!'
    
        self.embeddings_layer = nn.Sequential(
            nn.Conv2d(len(CHAR_SET), self.embeddings_dim, 1),
            nn.Tanh())
        
        self.res_block = ResidualBlock(self.embeddings_dim)
        
        self.upsampling_layer1 = nn.Sequential(
                nn.ConvTranspose2d(self.embeddings_dim, 128, 4, stride=2),
                nn.ReLU())
        self.upsampling_layer2 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2),
                nn.ReLU())
        self.upsampling_layer3 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, stride=2),
                nn.ReLU())
        self.upsampling_layer4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, stride=2),
                nn.ReLU())
        self.upsampling_layer5 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, 4, stride=2),
                nn.ReLU())
        self.upsampling_layer6 = nn.Sequential(
            nn.Upsample(img_dim[1]),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.ReLU())
        
        self.change_res_block = ResidualBlock(3)
        self.activation_res_block = ResidualBlock(1)
        
        self.change_conv_layer = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.Tanh())
        self.activation_conv_layer = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid())
         
    def forward(self, descriptions):
        z = self.embeddings_layer(descriptions)
        z = self.res_block(z)
        z = self.upsampling_layer1(z)
        z = self.upsampling_layer2(z)
        z = self.upsampling_layer3(z)
        z = self.upsampling_layer4(z)
        z = self.upsampling_layer5(z)
        z = self.upsampling_layer6(z)
        
        change_z = z[:, :3]
        activation_z = z[:, 3:4]
        
        change_z = self.change_res_block(change_z)
        change_z = self.change_conv_layer(change_z)
        
        activation_z = self.activation_res_block(activation_z)
        activation_z = self.activation_conv_layer(activation_z)
        
        output = change_z * activation_z
        
        return output


def make_dataset(path, img_dim):
    crop_size = img_dim[1]
    transform = transforms.Compose([
       transforms.Resize(crop_size),
       transforms.CenterCrop(crop_size),
       transforms.ToTensor()
    ])

    return ImgDataSet(path, transform)

def tensor_to_img(tensor):
    tensor = tensor.squeeze()
    tensor = rearrange(tensor, 'c h w -> h w c')
    arr = tensor.detach().cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img
    

class AgentController():
    def __init__(
        self,
        img_dim = (3, 256, 256),
        encoder_path = './models/encoder_final.model',
        decoder_path = './models/decoder_final.model',
        data_dir = './data/'):
        
        self.dataset = make_dataset(data_dir, img_dim)
        
        self.encoder = torch.load(encoder_path)
        self.decoder = torch.load(decoder_path)
        
        self.orig_img = None
        self.gen_img = None
        
    def get_new_img(self):
        self.dataset.shuffle()
        data_iter = self.dataset.get_batch_iter(1)
        self.orig_img = next(data_iter).cuda()
        self.gen_img = torch.zeros_like(self.orig_img)

        return tensor_to_img(self.orig_img)
        
    def step_gen_img(self):
        logits = self.encoder(self.orig_img, self.gen_img)
        self.gen_img = torch.clamp(self.decoder(logits) + self.gen_img.detach(), 0, 1)
        
        oned_logits = rearrange(logits.squeeze(), 'c h w -> (h w) c')
        char_indices = torch.argmax(oned_logits, dim=1).detach().cpu().numpy()
        text = ''.join([CHAR_SET[ci] for ci in char_indices])
        
        return tensor_to_img(self.gen_img), text


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ac = AgentController()

    for i in range(2):
        plt.imshow(ac.get_new_img())
        plt.show()

        for j in range(4):
            plt.imshow(ac.step_gen_img())
            plt.show()