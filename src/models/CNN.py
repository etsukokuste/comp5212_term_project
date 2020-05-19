import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #This CNN does not reshapes the last axis to first axis
        #This forward function does not require the gound truth label Pos (Since no rehaoing is being done)
        #Input to CNN : batch_size*C*H*W=(batch_size*10*56*924)
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=5, padding=2)  # (BS,5,56,924)--->(BS,64,56,924)
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4))  # (BS,64,56,924)--->(BS,64,56,231)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)  # (BS,64,56,231)--->(BS,32,56,231)
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 4))  # (BS,32,56,231)--->(BS,32,56,57)

        self.fc1 = nn.Linear(32 * 56 * 57, 512)  # (BS,32*56*57)--->(BS,512)
        self.fc2 = nn.Linear(512, 256)  # (BS,512)--->(BS,256)
        self.fc3 = nn.Linear(256, 128)  # (BS,256)--->(BS,128)
        self.fc4 = nn.Linear(128, 3)   # (BS,128)--->(BS,3)

    def forward(self, h_re, h_im, snr):
        #This function concatenates the input h_re, h_im and passes it to the CNN
        #It outputs the predicted Position out
        H = torch.sqrt(h_re ** 2 + h_im ** 2)
        # H=torch.cat((h_re, h_im), dim=-1)
        H = H.permute(0, 3, 1, 2)

        out = F.relu(self.conv1(H))
        out = self.avgpool1(out)
        out = F.relu(self.conv2(out))
        out = self.avgpool2(out)
        out = out.view(-1,32* 56 * 57)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
    
    def name(self):
        return "CNN"
