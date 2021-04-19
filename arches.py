# architectures
import torch.nn as nn
import torch.nn.functional as F
class Autoencoder(nn.Module):
    def __init__(self, z_dim):
        super(Autoencoder, self).__init__()
        ## Encoding: Unconditional samples
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1) # Input: (bs, 3, img_size, img_size)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias = False)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias = False)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias = False)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1, bias = False)
        self.conv5_bn = nn.BatchNorm2d(1024)
        self.conv6 = nn.Conv2d(1024, 2048, 4, 2, 1, bias = False)
        self.conv6_bn = nn.BatchNorm2d(2048)
        
        #self.conv7 = nn.Conv2d(2048, z_dim, 4, 2, 0) # Output: (bs, c_dim, 1, 1)
        self.fce = nn.Linear(2048, z_dim)

        ## Decoding:
        self.fcd = nn.Linear(z_dim, 2048)
        
        #self.deconv1 = nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias = False) # Not sure how this looks
        #self.deconv1_bn = nn.BatchNorm2d(1024)
        
        self.deconv2 = nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias = False)
        self.deconv2_bn = nn.BatchNorm2d(1024)
        self.deconv3 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias = False)
        self.deconv3_bn = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False)
        self.deconv4_bn = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False)
        self.deconv5_bn = nn.BatchNorm2d(128)
        self.deconv6 = nn.ConvTranspose2d(128, 64, 4, 3, 1, bias = False)
        self.deconv6_bn = nn.BatchNorm2d(64)
        self.deconv7 = nn.ConvTranspose2d(64, 3, 6, 2, 1, bias = False)


    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def encode(self, x):
        # Encode data x to 2 spaces: condition space and variance-space
        #print('enter encode', x.shape)
        x = F.relu(self.conv1(x), 0.2)
        #print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #print(x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print(x.shape)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        #print(x.relu)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        #print(x.shape)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        #print(x.shape)
        #z = self.conv7(x)
        #print(x.shape)
        z = self.fce(x.squeeze())
        #print(z.shape)
        #z = torch.sigmoid(self.conv5(x)) # Variance-space unif~[0,1]
        #print(z.shape)
        return z

    def decode(self, z):
        #print('enter decode', z.shape)
        #x = self.deconv1_bn(self.deconv1(z))
        #print(z.shape)
        x = F.relu(self.fcd(z)).unsqueeze(-1).unsqueeze(-1)
        #print(x.shape)
        #print(x.shape)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        #print(x.shape)
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        #print(x.shape)
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        #print(x.shape)
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        #print(x.shape)
        x = F.relu(self.deconv6_bn(self.deconv6(x)))
        #print(x.shape)
        x = self.deconv7(x)
        return torch.tanh(x)

    def forward(self, x):
        return self.decode(self.encode(x))