import torch

class RecognitionModule:
    def __init__(self, repo_path, model, device):
        # create the image recognition yolo model
        torch.device(device)
        self.model = torch.hub.load(repo_path, 'custom', path=model, source="local")

    def detect(self, image):
        return self.model(image)