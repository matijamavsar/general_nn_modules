import torch.nn as nn
import torch
from torch.nn.modules.activation import MultiheadAttention
import torchvision


class IntentNet(nn.Module):
    
    def __init__(self, use_googlenet=False):
        super(IntentNet, self).__init__()
        
        if use_googlenet:
            print('Using GoogleNet as CNN')
            self.cnn = torchvision.models.googlenet(pretrained=True)
            self.cnn.fc = nn.Identity()
            self.cnn.conv1.conv = nn.Conv2d(4, 64, kernel_size=(7, 7), 
                                            stride=(2, 2), padding=(3, 3), bias=False)
            self.cnn = nn.Sequential(*list(self.cnn.children()))
    
        # print('Using ResNet as CNN')
        # self.cnn = torchvision.models.resnet18(pretrained=True)
        # self.cnn.fc = nn.Identity()
        # self.cnn.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), 
        #                           stride=(2, 2), padding=(3, 3), bias=False)
        
        else:
            print('Using custom CNN')
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 32, 4), nn.GroupNorm(32, 32),
                nn.ReLU(), nn.MaxPool2d(3, 3),
                nn.Conv2d(32, 128, 2), nn.GroupNorm(128, 128),
                nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 2), nn.GroupNorm(256, 256),
                nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 384, 2), nn.GroupNorm(384, 384),
                nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(384, 512, 3), nn.GroupNorm(512, 512),
                nn.ReLU(), nn.MaxPool2d(2, 2))
        
        dummy_in = torch.zeros(1, 4, 120, 160)
        dummy_out = self.cnn(dummy_in)
        linsize = dummy_out.reshape(-1).shape
        
        self.pretracker = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(linsize[0], 2500), nn.ReLU(),
            nn.LayerNorm(2500), 
            nn.Dropout(0.3), nn.Linear(2500, 400),
            nn.ReLU(), nn.LayerNorm(400)
        )

        self.tracker = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(400, 200), nn.ReLU(),
            nn.LayerNorm(200), nn.Dropout(0.2),
            nn.Linear(200, 7))
        
        self.pre_rnn = nn.Sequential(
            nn.Dropout(0.1),
        )
        
        self.rnn = nn.Sequential(
            nn.LSTM(400, 300, num_layers=2, batch_first=True),
        )
        
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(300),
            nn.Linear(300, 4), # nn.LogSoftmax(dim=-1), <-- Done using loss function
        )
        
    def forward(self, input):
        dims = input.shape
        input = input.view(-1, dims[2], dims[3], dims[4])
            
        out_pretracker = self.cnn(input)
        out_pretracker = out_pretracker.view(dims[0], dims[1], -1)
        out_pretracker = self.pretracker(out_pretracker)
        out_tracker = self.tracker(out_pretracker)
        out_demo = self.pre_rnn(out_pretracker)
        out_d0 = self.rnn[0](out_demo)[0]
        # out_d1 = self.rnn[1](out_demo)[0]
        # out_d2 = self.rnn[2](out_demo)[0]
        # out_demo = self.output(torch.cat((out_d0, out_d1, out_d2), axis=-1))
        out_demo = self.output(out_d0)

        return out_demo, out_tracker
    
    def forward_tracker(self, input):
        dims = input.shape
            
        out_pretracker = self.cnn(input)
        out_pretracker = out_pretracker.view(dims[0], -1)
        out_pretracker = self.pretracker(out_pretracker)
        out_tracker = self.tracker(out_pretracker)

        return out_tracker


class OptiNet(nn.Module):
    
    def __init__(self):
        super(OptiNet, self).__init__()
                
        self.pretracker = nn.Sequential(
            nn.Linear(3, 2500), nn.ReLU(),
            nn.LayerNorm(2500), 
            nn.Dropout(0.3), nn.Linear(2500, 400),
            nn.ReLU(), nn.LayerNorm(400)
        )
        
        self.pre_rnn = nn.Sequential(
            nn.Dropout(0.1),
        )
        
        self.rnn = nn.Sequential(
            nn.LSTM(400, 300, num_layers=2, batch_first=True),
        )
        
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(300),
            nn.Linear(300, 4), # nn.LogSoftmax(dim=-1), <-- Done using loss function
        )
        
    def forward(self, input):
            
        out_pretracker = self.pretracker(input)
        out_demo = self.pre_rnn(out_pretracker)
        out_d0 = self.rnn[0](out_demo)[0]
        # out_d1 = self.rnn[1](out_demo)[0]
        # out_d2 = self.rnn[2](out_demo)[0]
        # out_demo = self.output(torch.cat((out_d0, out_d1, out_d2), axis=-1))
        out_demo = self.output(out_d0)

        return out_demo
    

class RIMEDNet(nn.Module):
    
    def __init__(self, use_pretrained=False):
        super(RIMEDNet, self).__init__()
        
        # TODO: also add Inception-v4
        if use_pretrained=="googlenet":
            print('Using GoogleNet as CNN')
            self.cnn = torchvision.models.googlenet(pretrained=True)
            self.cnn.fc = nn.Identity()
            self.cnn.conv1.conv = nn.Conv2d(4, 64, kernel_size=(7, 7), 
                                            stride=(2, 2), padding=(3, 3), bias=False)
            self.cnn = nn.Sequential(*list(self.cnn.children()))
        elif use_pretrained=='alexnet':
            print('Using AlexNet')
            self.cnn = torchvision.models.alexnet(pretrained=True)
            del self.cnn.classifier[-1]
            del self.cnn.classifier[-1]
            depth_conv = torch.cat((torch.zeros((64, 1, 11, 11)), 
                                    self.cnn.features[0].weight), axis=1)
            self.cnn.features[0].weight = nn.parameter.Parameter(depth_conv)
        else:
            print('Using custom CNN')
            self.cnn = nn.Sequential(
                # TODO: probaj brez GroupNorm!
                nn.Conv2d(4, 32, 6), nn.GroupNorm(32, 32),
                nn.ReLU(), nn.MaxPool2d(3, 3),
                nn.Conv2d(32, 64, 5), nn.GroupNorm(64, 64),
                nn.ReLU(), nn.MaxPool2d(3, 3),
                nn.Conv2d(64, 256, 4), nn.GroupNorm(256, 256),
                nn.ReLU(), nn.MaxPool2d(3, 3))

                # nn.Conv2d(4, 32, 4), nn.GroupNorm(32, 32),
                # nn.ReLU(), nn.MaxPool2d(3, 2),
                # nn.Conv2d(32, 64, 4), nn.GroupNorm(64, 64),
                # nn.ReLU(), nn.MaxPool2d(3, 4),
                # nn.Conv2d(64, 256, 4), nn.GroupNorm(256, 256),
                # nn.ReLU(), nn.MaxPool2d(4, 4))

        dummy_in = torch.zeros(1, 4, 227, 227)
        dummy_out = self.cnn(dummy_in)
        linsize = dummy_out.reshape(-1).shape
        print('Size of first linear layer:', linsize)

        self.hidden_size = 300
        
        self.fc1 = nn.Sequential(
            nn.Linear(linsize[0], 2200), nn.ReLU(), # nn.LayerNorm(2200),
            nn.Linear(2200, 1000), nn.ReLU(), # nn.LayerNorm(1000),
            )
        
        self.rnn = nn.Sequential(
            nn.LSTM(1000, self.hidden_size, num_layers=2, batch_first=True))
        
        self.output = nn.Sequential(
            nn.Linear(self.hidden_size, 100), nn.ReLU(), # nn.LayerNorm(100),
            nn.Linear(100, 190))
        
    def forward(self, input):
        dims = input.shape
        # hidden = (
        #     torch.ones(2, dims[0], self.hidden_size).type_as(input),
        #     torch.ones(2, dims[0], self.hidden_size).type_as(input),
        #     )

        input = input.view(-1, dims[2], dims[3], dims[4])
        out = self.cnn(input)
        
        # out = []
        # for i_frame in range(input.shape[1]):
        #     out.append(self.cnn(input[:, i_frame]))
        # out = torch.stack(out).permute(1, 0, 2, 3, 4)

        out = out.view(dims[0], dims[1], -1)
        out = self.fc1(out)
        # out = self.rnn[0](out, hidden)[0]
        out = self.rnn[0](out)[0]
        out = self.output(out)

        return out