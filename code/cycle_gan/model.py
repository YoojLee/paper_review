import torch
from torch import nn
from collections import OrderedDict

shape_dict=dict() # for checking the output's shape

class ResidualBlock(nn.Module):
    """
    residual block (He et al., 2016)
    """
    def __init__(self, in_channels:int, out_channels:int):
        """
        - Args
            in_channels: number of channels for an input feature map
            out_channels: number of channels for an output feature map

        - Note
            fixed a kernel_size to 3
        """
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x):
        output = self.block(x) + x # skip-connection

        return output


class Generator(nn.Module):
    def __init__(self, init_channel:int, kernel_size:int, stride:int, n_blocks:int=6):
        """
        - Args
            stride: 2
            f_stride: 1/2
            kernel size: 9 (first and last), 3 for the others

            3 convolutions & 6 residual_blocks, 2 fractionally-strided convolutions
            one convolutions (features to RGB) -> 3 channel로 보낸다.

            instance normalization -> non-residual convolutional layers

            non-residual convolutional layers: followed by spatial batch normalization
            relu nonlinearities with the exception of the output layer
            + a scaled tanh to ensure that the output image has pixels in the range [0,255]
        """
        super(Generator, self).__init__()
        
        self.init_channel=init_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.n_blocks=n_blocks

        layers = OrderedDict()
        layers['conv_first'] = self._make_block(in_channels=3, out_channels=self.init_channel, kernel_size=7, stride=1, padding='same') # first layer

        # downsampling path (d_k) -> two downsampling blocks
        for i in range(2):
            ic = self.init_channel*(i+1)
            k = 2*ic
            layers[f'd_{k}'] = self._make_block(in_channels=ic, out_channels=k, kernel_size=self.kernel_size, stride=self.stride)

        # residual block (R_k) -> 6 or 9 blocks
        for i in range(self.n_blocks):
            layers[f'R{k}_{i+1}'] = ResidualBlock(k, k) # in_channel = out_channel로 동일한 channel dimension 유지

        # upsampling path (u_k) -> two upsampling blocks
        for i in range(2):
            k = int(k/2)
            layers[f'u_{k}'] = self._make_block(in_channels=k*2, out_channels=k, kernel_size=self.kernel_size, stride=self.stride, mode='u')

        # last conv layer
        layers['conv_last'] =  self._make_block(in_channels=self.init_channel, out_channels=3, kernel_size=7, stride=1, padding='same') # last conv layer (to rgb)
        #layers['sigmoid'] = nn.Sigmoid()
        layers['tanh'] = nn.Tanh()

        self.model = nn.Sequential(
            layers
        )
        
    def forward(self, x):
        op = self.model(x)
        assert op.shape == x.shape, f"output shape ({op.shape}) must be same with the input size ({x.shape})"
        return op

    def _make_block(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int=1, mode:str='d'):
        """
        builds a conv block

        - Args
            in_channels (int): # of channels of input feature map
            out_channels (int): # of channels of output feature map
            kernel_size (int): kernel size for a convolutional layer
            stride (int): stride for a convolution
            padding (int): an amount of padding for input feature map
            mode (str): 'd'(downsampling mode) or 'u'(upsampling mode) (default: 'd')
        """
        
        block = []
        if mode.lower() == 'd':
            block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode='reflect'))

        elif mode.lower() == 'u':
            block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=1)) # output size를 input이랑 같게 해주려면 이렇게 설정을 해줄 수밖에 없음.

        block += [nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True)]

        return nn.Sequential(*block)



class Discriminator(nn.Module):
    def __init__(self, n_layers:int=4, input_c:int=3, n_filter:int=64, kernel_size:int=4):
        """
        - Args
            n_layers (int): number of convolutional layers in the network (default=3)
            input_c (int): number of input channels (default=3)
            n_filter (int): number of filters in the first convolutional layer (default=64)
            kernel_size (int): kernel size for every convolutional layers in the network (default=4)
        
        - Output
            2-D tensor (b,)
        
        PatchGAN 구조 사용 -> 이미지를 patch로 넣어주겠다는 것이 아님. receptive field를 이용하는 듯함.

        size of receptive fields = 1 + L(K-1); L = # of layers, K = kernel size (under the stride=1)
        이 receptive fields를 70으로 잡아주겠다.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential()
        self.kernel_size=kernel_size
        self.n_layers = n_layers
        layers = []
        
        # building conv block
        for i in range(self.n_layers):
            if i==0:
                ic, oc = input_c, n_filter
                layers.append(self._make_block(ic, oc, kernel_size=self.kernel_size, stride=2, padding=1, normalize=False))
            else:
                ic = oc
                oc = 2*ic
                stride=2
                
                if i == self.n_layers-1: # 마지막 레이어(c512)의 경우, stride=1로 설정할 것.
                    stride=1

                layers.append(self._make_block(ic, oc, kernel_size=self.kernel_size, stride=stride, padding=1))

        # prediction
        layers.append(nn.Conv2d(oc, 1, kernel_size=self.kernel_size, stride=1, padding=1))

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)


    def _make_block(self, in_channels, out_channels, stride, kernel_size=3, padding=0, normalize=True):
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size, padding=padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

# check the output size within a model -> dict key를 어떻게 원하는 형식으로 바꿀 수 있을지 생각해볼 것.

# hook function은 forward 이후에 activate된다는 점을 잊지 말자.
def hook_fn(m, _, o):
    """
    m: module
    i: input
    o: output
    """
    shape_dict[m]=o.shape
    

def get_all_layers(net:nn.Module, hook_fn=hook_fn):
    for name, layer in net._modules.items():
        #print(name)
        if isinstance(layer, nn.Sequential):
            get_all_layers(layer)
        else:
            layer.register_forward_hook(hook_fn)


if __name__ == "__main__":
    kwargs = {
        'init_channel': 64,
        'stride': 2,
        'kernel_size': 3,
        'n_blocks': 9
    }
    ip = torch.randn(1,3,256,256)

    D = Discriminator()
    G = Generator(**kwargs)
    
    get_all_layers(D) # forward hook to check the output shape of the feature map after every layer.

    op_d = D(ip) # forward
    op_g = G(ip)
    print(op_g.shape)
    # print(*shape_dict.values(), sep="\n")