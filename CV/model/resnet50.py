import torch
from torch import nn



### BottleNeck Block
class BottleNeck(nn.Module):
    '''
    - 1x1, 3x3, 1x1 convolution layer가 순차적으로 쌓이는 구조
    - 각 convolution layer와 activation layer 사이에 batch normalization 적용
    - expansion 변수: 출력 차원 수 증가 → 더 많은 특징 학습 및 성능 향상
    '''

    expansion = 4 # ResNet에서 병목 블록을 정의하기 위한 hyperparameter

    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        ## 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        ## 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        ## 1x1 conv
        # 다음 계층의 in_channels와 이전 계층의 out_channels가 일치하도록 조정해주기 위해
        # self.expansion(= 4) * out_channels 수행
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        
        ## activation fn(for non-linearity)
        # ReLU 적용
        self.relu = nn.ReLU(inplace = True)

        ## DownSampling
        # stride = 2이거나 in/out_channels가 다를 경우 적용
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    
    def forward(self, x):
        i = x # identity mapping(skip connection)을 위한 original input 저장

        ## Conv Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # 마지막 계층에는 activation 적용 x


        ## DownSampling
        if self.downsample is not None:
            i = self.downsample(i)
        

        ## Identity Mapping(Residual Connection)
        x += i
        x = self.relu(x) # identity mapping 후 activation 적용


        return x
    


### ResNet50 Model
class ResNet50(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        self.in_channels = 64  # 초기 입력 채널

        ## conv1
        # 7x7 Convolution + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)

        ## conv2
        # 3x3 maxpooling + residual block(64, 64, 64*4) x 3
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self.make_layer(BottleNeck, 64, 3)

        ## conv3
        # residual block(128, 128, 128*4) x 4
        self.layer2 = self.make_layer(BottleNeck, 128, 4, stride = 2)

        ## conv4
        # residual block(256, 256, 256*4) x 6
        self.layer3 = self.make_layer(BottleNeck, 256, 6, stride = 2)

        ## conv5
        # residual block(512, 512, 512*4) x 3
        self.layer4 = self.make_layer(BottleNeck, 512, 3, stride = 2)

        ## Output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleNeck.expansion, num_classes)


    ## ResNet의 각 Stage를 구성하는 블록을 생성하는 함수     
    def make_layer(self, block, out_channels, num_blocks, stride = 1):
        layers = []

        # 첫 번째 블록에서는 downsampling 적용
        # identity mapping 시 feature map의 형태를 맞춰주기 위함
        layers.append(block(self.in_channels, out_channels, stride, downsample = True))
        # 채널 업데이트
        self.in_channels = out_channels * block.expansion

        # 나머지 블록들은 downsampling 없이 추가
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        ## conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        ## conv2
        x = self.maxpool(x)
        x = self.layer1(x)

        ## conv3
        x = self.layer2(x)

        ## conv4
        x = self.layer3(x)

        ## conv5
        x = self.layer4(x)
        
        ## Output layer
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)  # flatten
        x = self.fc(h)

        return x