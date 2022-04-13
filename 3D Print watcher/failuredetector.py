import torch
import torch.nn as nn # neural network
import torch.nn.functional as F # raw function


class Net(nn.Module):
    """
    The neural network structure
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)


class FailureDetector:
    def __init__(self, model_path):
        self.model = Net()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def detect(self, array):
        """
        determind if the data passed constitute a failed print

        :param array: 20 x 6 array from the yolo recog module
        :return: signmoidal tensor coorsponding to print status
        """
        array = torch.tensor(array, dtype=torch.float32)
        array = torch.flatten(array)
        return self.model(array)
