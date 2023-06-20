import torch

model = torch.hub.load("WongKinYiu/yolov7", 'custom','yolov7.pt')

model.eval()