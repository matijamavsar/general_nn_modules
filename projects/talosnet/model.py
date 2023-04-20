import torch.nn as nn
import torch
import torchvision

class TalosNet(nn.Module):
    
    def __init__(self, rnn_type='LSTM', use_residual=False):
        super(TalosNet, self).__init__()

        self.use_residual = use_residual
                
        self.fc_traj = nn.Sequential(
            nn.Linear(3, 2500), nn.ReLU(),
            nn.LayerNorm(2500), 
            # nn.Dropout(0.3), 
            nn.Linear(2500, 400),
            nn.ReLU(), nn.LayerNorm(400)
        )

        self.pre_rnn = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Identity(),
        )

        if rnn_type == 'LSTM':
            self.rnn = nn.Sequential(
                nn.LSTM(400, 400, num_layers=1, batch_first=True),
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.Sequential(
                nn.GRU(400, 400, num_layers=1, batch_first=True),
            )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(400),
            nn.Linear(400, 3), # nn.LogSoftmax(dim=-1), <-- Done using loss function
        )
        
    def forward(self, input):
            
        out_fc_traj = self.fc_traj(input)
        out_pos = self.pre_rnn(out_fc_traj)
        out_d0 = self.rnn(out_pos)[0]
        if self.use_residual:
            out_pos = self.output(out_d0 + out_pos)
        else:
            out_pos = self.output(out_d0)

        return out_pos


class TalosDepNet(nn.Module):
    
    def __init__(self, rnn_type='LSTM', use_residual=False):
        super(TalosDepNet, self).__init__()

        self.use_residual = use_residual
        self.cnn = torchvision.models.googlenet(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.cnn.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                                        stride=(2, 2), padding=(3, 3), bias=False)
        self.cnn = nn.Sequential(*list(self.cnn.children()))

        dummy_in = torch.zeros(1, 1, 224, 224)
        dummy_out = self.cnn(dummy_in)
        linsize = dummy_out.reshape(-1).shape

        self.prejoin = nn.Sequential(
            nn.Linear(linsize[0], 3), nn.ReLU(),
            nn.LayerNorm(3), 
        )

        self.fc_joined = nn.Sequential(
            nn.Linear(6, 2500), nn.ReLU(),
            nn.LayerNorm(2500), 
            # nn.Dropout(0.3), 
            nn.Linear(2500, 400),
            nn.ReLU(), nn.LayerNorm(400)
        )

        self.pre_rnn = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Identity(),
        )

        if rnn_type == 'LSTM':
            self.rnn = nn.Sequential(
                nn.LSTM(400, 400, num_layers=1, batch_first=True),
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.Sequential(
                nn.GRU(400, 400, num_layers=1, batch_first=True),
            )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(400),
            nn.Linear(400, 3), # nn.LogSoftmax(dim=-1), <-- Done using loss function
        )
        
    def forward(self, input_img, input_traj):
        out_cnn = self.cnn(input_img)
        out_cnn = self.prejoin(out_cnn)
        out_fc_joined = self.fc_joined(torch.cat((input_traj, out_cnn), axis=-1))
        out_pos = self.pre_rnn(out_fc_joined)
        out_d0 = self.rnn(out_pos)[0]
        if self.use_residual:
            out_pos = self.output(out_d0 + out_pos)
        else:
            out_pos = self.output(out_d0)

        return out_pos