import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(56 * 924 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 3)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, h_re, h_im, snr):

        H = torch.sqrt(h_re**2 + h_im**2)  #(batch size , size, 56, 924, 5)
        H = H.reshape(H.shape[0], H.shape[1]* H.shape[2]*H.shape[3])

        out = self.dropout1(F.relu(self.fc1(H)))
        out = self.dropout2(F.relu(self.fc2(out)))
        out = self.dropout3(F.relu(self.fc3(out)))
        out = self.fc4(out)

        return out

