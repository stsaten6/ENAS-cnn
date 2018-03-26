import numpy as np
from collections import defaultdict, deque
import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import models.shared_base
from utils import get_logger, get_variable, keydefaultdict
import copy

# logger = get_logger()


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def conv1x3x1(in_planes, out_planes, stride=1):
    #TODO 1x3 3x1 kernel size
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = (1,3), stride=stride,
                    padding=(0, 1), bias=False),
            nn.Conv2d(out_planes, out_planes, kernel_size = (3,1), stride=stride,
                    padding=(1, 0), bias=False),
    )


def isnan(tensor):
    return np.isnan(tensor.cpu().data.numpy()).sum() > 0


def conv(kernel, in_planes, out_planes):
    if kernel == '1x1':
        _conv = conv1x1
    elif kernel == '3x3':
        _conv = conv3x3
    elif kernel == '1x3x1':
        _conv = conv1x3x1
    elif kernel == '5x5':
        _conv = conv5x5
    else:
        raise NotImplementedError(f"Unknown kernel size: {kernel}")
    return nn.Sequential(
             _conv(in_planes, out_planes),
             nn.BatchNorm2d(out_planes),
             nn.ReLU(inplace=True),
    )



class CNN(models.shared_base.SharedModel):
    """Shared CNN model."""
    def __init__(self, args, images):
        # super(CNN, self).__init__()
        models.shared_base.SharedModel.__init__(self)
        self.args = args
        self.images = images  #imagedataset
        self.channel = self.args.cnn_channel #[3, 128, 256, 512, 1024， 1024]
        in_planes = 3
        out_planes = 128
        self.conv = defaultdict(dict)
        self.channel_bridge = defaultdict(dict)
        #self.args.cnn_num_blocks --> [3, 3, 3, 3]
        #NOTE create shared convolution
        for layer_idx, layer_num in enumerate(self.args.cnn_num_blocks):
        #TODO default dict useage change mode
            for block_idx in range(layer_num):
                # self.conv[layer_idx][block_idx] = list()
                if block_idx == 0:
                    out_planes = self.args.cnn_channel[layer_idx + 1]
                # for jdx, cnn_type in enumerate(self.args.shared_cnn_types):
                #     self.conv[layer_idx][block_idx].append(conv(cnn_type, in_planes, out_planes))
                abs_layer_idx = block_idx + sum(self.args.cnn_num_blocks[:layer_idx])
                for jdx, cnn_type in enumerate(self.args.shared_cnn_types):
                    self.conv[abs_layer_idx][cnn_type] = conv(cnn_type, in_planes, out_planes)
                in_planes = out_planes
            for previous_idx in range(layer_idx + 1):
                self.channel_bridge[layer_idx][previous_idx] = conv('1x1',
                                                                    self.channel[previous_idx],
                                                                    self.channel[layer_idx+1])
        self._conv = nn.ModuleList([self.conv[idx][jdx]
                                   for idx in self.conv
                                   for jdx in self.conv[idx]])
        self._channel_bridge = nn.ModuleList([self.channel_bridge[idx][jdx]
                                              for idx in self.channel_bridge
                                              for jdx in self.channel_bridge[idx]])
        print(self.conv)
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        # self.predict = torch.nn.Conv2d(1024,10, 1)# for classification
        #TODO change downsampling from avgpool to bilinear
        self.avgpool = torch.nn.AvgPool2d(kernel_size = 4, stride=1)
        self.downsample = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        #use xavier to initialize parameter
        self.predict = torch.nn.Linear(1024, 10)
        
        self.reset_parameters()
        print(f'# of parameters: {format(self.num_parameters, ",d")}')

    def forward(self,
                inputs,
                dag,
                is_train = True):
        '''
        args:
            inputs : image dataset [batches, channel, w, h]
            dag : dag that show use what kind of conv and use which layer to skip connect
            self.conv[0,1,2 ... 9, 10, 11]
        '''
        #TODO when to chagne is_train
        # inputs = Variable(inputs)
        batch_size = inputs.size(0)
        is_train = is_train and self.args.mode is ['train']
        #[img, layer1, layer2, layer3, (maxpool) layer4, ... layer 12] so with length of 13
        results = list()
        #TODO how to save memory?
        results.append(inputs)
        #use to record whtn do maxpooling
        time_to_maxpool = copy.deepcopy(self.args.cnn_num_blocks)
        time_to_maxpool[-1]+=1
        time_to_maxpool.reverse()

        #用time_to_maxpool 来记录什么时候要maxpooling
        #[4,3,3,3]  每次第一个减1 当第一个为0 时代表到头了需要做maxpooling 了 最后一次不用maxpooling
        #TODO too complicated and need to optimization
        #the last node is avgpooling
        #dag 0 is None
        # print(dag)
        # print(type(dag))
        for idx in range(sum(self.args.cnn_num_blocks)+1):
            # print("---new---,idx:", idx)
            if idx == 0:
                continue
            layer = dag[idx+1][0]
            x = results[-1]


            if idx == 1:
                func_name = dag[idx][0][1]
                # print(func_name)
                # print(x.size())
                # print(self.conv[idx-1][func_name])
                x = self.conv[idx-1][func_name](x)
                if time_to_maxpool[-1] != 0:
                    time_to_maxpool[-1] -=1
                else:
                    x = self.maxpool(x)
                    time_to_maxpool.pop()
                    time_to_maxpool[-1] -=1
                results.append(x)
                # print("x.size())
            else:
                # print(prev_node)
                prev_node = dag[idx][0][0]
                prev_data = results[prev_node]
                # print("prev_node", prev_node)
                # print("prev_data size", prev_data.size())
                if int((int(prev_node) - 1)//3) != int((idx - 1 - 1)//3):
                    #TODO test upsample can use for down sample?
                    #TOTEST list 取出来是否还可以反向传播
                    # prev_data = F.upsample(prev_data, scale = pow(2, idx/3 - prev_node/3))
                    # print("channel_conv")
                    for _ in range(int((idx - 1 - 1)//3) - int(abs(int(prev_node) - 1)//3)):
                        # print("downsample-----")
                        # print("downsample-----", prev_data.size())
                        prev_data = self.downsample(prev_data)
                    #channel with 0 need to more comment
                    # print(int((idx - 1)//3), int((int(prev_node)-1)//3 + 1) )
                    prev_data = self.channel_bridge[int((idx - 1 - 1)//3)][int((int(prev_node) -1)//3 + 1)](prev_data)
                    # print(channel_conv)
                    # prev_data = channel_conv(prev_data)
                    # print(prev_data.size())
                # x = results[-1] + result[prev_node + 1]
                # print("idx, prev_node", idx, prev_node)
                # print("prev_data size:", prev_data.size())
                # print("x size:", x.size())
                x += prev_data
                func_name = dag[idx][0][1]
                x = self.conv[idx-1][func_name](x)
                if time_to_maxpool[-1] != 0:
                    time_to_maxpool[-1] -=1
                else:
                    # print("maxpooling, " "idx----", idx)
                    x = self.maxpool(x)
                    time_to_maxpool.pop()
                    time_to_maxpool[-1] -=1
                results.append(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = x.cuda()
        x = self.predict(x)
        # print(x.size())
        return x

    def get_f(self, name):
        #TODO activation are used for rnn
        name = name.lower()
        if name == 'relu':
            f = F.relu
        elif name == 'tanh':
            f = F.tanh
        elif name == 'identity':
            f = lambda x: x
        elif name == 'sigmoid':
            f = F.sigmoid
        elif name == 'upsample':
            f = F.upsample
        # elif name == 'downsample':
        #     f = F.upsample
        else:
            return None
        return f

    def get_num_cell_parameters(self, dag):
        '''
        use for what?
        pass
        maybe useless
        '''
        pass
    def reset_parameters(self):
        #TODO how to initialize when convolution with bias
        '''
        initialize the paramerter
        ues the method in resenet
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L112-L118
        '''
        
        for m in self.modules():
            #m.cuda()
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #for param in self.parameters():
        #    torch.nn.init.xavier_uniform(param)
