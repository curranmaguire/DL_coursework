import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from IPython import display as disp
import time
import os
from cleanfid import fid
from torchvision.utils import save_image
import shutil

###########################PARAMS######################################
#######################################################################
IMG_SIZE = 32
BATCH_SIZE = 2**6
timesteps = 16
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

#helper function for unnormalizing data for viewing
def inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

class_names = ['apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',]

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])),
    batch_size=BATCH_SIZE, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])),
    batch_size=BATCH_SIZE, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))





#----------------------------------------------------------------------------------------
#---------------------------------MODEL OUTLINE-----------------------------------------
class DSC(nn.Module):
    '''https://arxiv.org/pdf/1704.04861.pdf
    depthwise seperable convolution. 
    a seperate conv is applied to each channel this is done using 
    groups = channels_in,
    then a pointwise is used to generate new channels
    '''
    
    def __init__(self, chan_in, chan_out, k=3, stride=1, pad=1):
        super(DSC, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(chan_in, chan_in, kernel_size = k, stride=stride, padding=pad, groups=chan_in),
            nn.BatchNorm2d(chan_in),
            nn.LeakyReLU()
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, kernel_size = 1, stride=1, padding=0),
            nn.BatchNorm2d(chan_out),
            nn.LeakyReLU()
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        
        self.r1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.in1 = nn.InstanceNorm2d(32)
        
        self.r2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64)
        
        self.r3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128)
        
        self.r4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.in4 = nn.InstanceNorm2d(128)
        
        self.r5 = nn.ReflectionPad2d(1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.in5 = nn.InstanceNorm2d(128)
        
        self.r6 = nn.ReflectionPad2d(1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3)
        self.in6 = nn.InstanceNorm2d(128)
        
        self.r12 = nn.ReflectionPad2d(1)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3)
        self.in12 = nn.InstanceNorm2d(128)
        
        self.uconv16 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in16 = nn.InstanceNorm2d(64)
        
        self.uconv17 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in17 = nn.InstanceNorm2d(32)
        
        self.r18 = nn.ReflectionPad2d(3)
        self.conv18 = nn.Conv2d(32, 3, kernel_size=7, stride=1)
        self.in18 = nn.InstanceNorm2d(3)
        
    def forward(self, input):
        
        x = F.leaky_relu(self.in1(self.conv1(self.r1(input))), negative_slope=0.2)
        x = F.leaky_relu(self.in2(self.conv2(self.r2(x))), negative_slope=0.2)
        x = F.leaky_relu(self.in3(self.conv3(self.r3(x))), negative_slope=0.2)
        x1 = F.leaky_relu(self.in4(self.conv4(self.r4(x))), negative_slope=0.2)
        x1 = F.leaky_relu(self.in5(self.conv5(self.r5(x1))), negative_slope=0.2)
        
        x = x + x1
        
        x1 = F.leaky_relu(self.in12(self.conv12(self.r12(x))), negative_slope=0.2)
      
        x = x + x1
        
        x = F.leaky_relu(self.in16(self.uconv16(x)), negative_slope=0.2)
        x = F.leaky_relu(self.in17(self.uconv17(x)), negative_slope=0.2)
        x = F.leaky_relu(self.in18(self.conv18(self.r18(x))), negative_slope=0.2)
        
        return torch.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(128)

        self.head = nn.Conv2d(128, 1, 4, padding=1)
                
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), negative_slope=0.2)
        x = F.leaky_relu(self.in2(self.conv2(x)), negative_slope=0.2)
        
        x = self.head(x)
        x = F.avg_pool2d(x, x.shape[2:])
        return torch.sigmoid(x).view(x.size(0), -1)


G = generator().to(device)
G_optim = torch.optim.Adam(G.parameters(), lr=0.0002)
G_params = len(torch.nn.utils.parameters_to_vector(G.parameters()))


D = Discriminator().to(device)
D_optim = torch.optim.Adam(D.parameters(), lr=0.0002)
D_params = len(torch.nn.utils.parameters_to_vector(D.parameters()))
criterion = nn.BCEWithLogitsLoss()
import sys
print(sys.version)
print(f'> Number of model parameters {G_params + D_params}\n')
if (G_params + D_params) > 1000000:
    print("> Warning: you have gone over your parameter budget and will have a grade penalty!")
steps = 0


#----------------------------------------------------------------------------------------
#---------------------------------MODEL TRAINING-----------------------------------------

# keep within our optimisation step budget
display_step = 5000
G_loss_arr = []
D_loss_arr = []
epoch = 0
while (steps < 50000):

    # arrays for metrics
    

    # iterate over some of the train dateset
    for i in range(display_step):
        img,_ = next(train_iterator)
        real = img.to(device)
        noise = torch.randn(img.shape, device=device)
        
        real_labels = torch.ones(BATCH_SIZE, 1).to(device)
        fake_labels = torch.zeros(BATCH_SIZE, 1).to(device)
        
        #trainD
        D_score_real = D(real).view(-1,1)
        D_score_fake = D(G(noise).detach()).view(-1,1)
        
        
        lossD_real = criterion(D_score_real, real_labels)
        lossD_fake = criterion(D_score_fake, fake_labels)
        
        loss_D = (lossD_real + lossD_fake)/2.0
        D_optim.zero_grad()
        loss_D.backward()
        D_optim.step()
        
        #---------------------------
        #train G
        
        fake = G(noise)
        D_score_fake = D(fake).view(-1,1)
        
        G_optim.zero_grad()
        loss_G = criterion(D_score_fake, real_labels)
        loss_G.backward()
        G_optim.step()
        
        steps += 1

        D_loss_arr.append([steps, loss_D.item()])
        G_loss_arr.append([steps, loss_G.item()])


    print('steps {:.2f}, D_loss: {:.3f}, G_loss: {:.3f}'.format(steps, loss_D.mean(), loss_G.mean()))

    # sample model and visualise results (ensure your sampling code does not use x)
    vis = torch.stack([inverse_normalize(real[0].cpu().detach()), inverse_normalize(fake[0].cpu().detach())])
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot images in the first subplot
    axs[0].imshow(torchvision.utils.make_grid(vis).cpu().numpy().transpose(1, 2, 0))
    axs[0].set_title('Real and Fake Images')
    axs[0].axis('off')  # Hide the axis on the image plot

    # Plot the loss graphs in the second subplot
    axs[1].plot([x[0] for x in D_loss_arr], [x[1] for x in D_loss_arr], '-', color='tab:grey', label="Discriminator Loss")
    axs[1].plot([x[0] for x in G_loss_arr], [x[1] for x in G_loss_arr], '-', color='tab:purple', label="Generator Loss")
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc="upper left")
    axs[1].set_title('Training Losses')

    # Display the figure
    plt.tight_layout()
    plt.savefig(f"training_fig_{epoch}")
    epoch += 1
    plt.close(fig=fig)
    G.train()


##################################################################################
################################# MODEL EVALUATION ###############################
# define directories
real_images_dir = 'real_images'
generated_images_dir = 'generated_images'
num_samples = 10000 # do not change

# create/clean the directories
def setup_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory) # remove any existing (old) data
    os.makedirs(directory)

setup_directory(real_images_dir)
setup_directory(generated_images_dir)

# generate and save 10k model samples
num_generated = 0
while num_generated < num_samples:

    # sample from your model, you can modify this
    z = torch.randn(img.shape).to(device)
    samples_batch = G(z).cpu().detach()

    for image in samples_batch:
        if num_generated >= num_samples:
            break
        save_image(image, os.path.join(generated_images_dir, f"gen_img_{num_generated}.png"))
        num_generated += 1

# save 10k images from the CIFAR-100 test dataset
num_saved_real = 0
while num_saved_real < num_samples:
    real_samples_batch, _ = next(test_iterator)
    for image in real_samples_batch:
        if num_saved_real >= num_samples:
            break
        save_image(image, os.path.join(real_images_dir, f"real_img_{num_saved_real}.png"))
        num_saved_real += 1

# compute FID
score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
print(f"FID score: {score}")