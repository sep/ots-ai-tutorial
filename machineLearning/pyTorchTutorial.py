import matplotlib.pyplot as plot
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

# TODO: This might be equivalent to nn.Sequential, if so, replace?
class TutorialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()
 
        self.fc3 = nn.Linear(6272, 64)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(64, 10)

    def chain(self,arg, functions):
        intermediate = arg
        for fn in functions:
            intermediate = fn(intermediate)
        return intermediate
        
    def forward(self, x):
        chain = [
            self.conv1,
            self.act1,
            self.drop1,
            self.conv2,
            self.act2,
            self.pool2,
            self.flat,
            self.fc3,
            self.act3,
            self.drop3,
            self.fc4
        ]
        return self.chain(x, chain)

    
def displayBatch(batch):
    images, labels = batch
    plot.imshow(images[0].numpy().squeeze(), cmap='gray_r');
    print(images[0].shape)
    plot.show()


def evaluate(model, testLoader, epoch=-1):
    acc = 0
    count = 0
    for inputs, labels in testLoader:
        predictions = model(inputs)
        acc += (torch.argmax(predictions, 1) == labels).float().sum()
        count += len(labels)
    acc /= count
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc * 100))


def trainIteration(model, lossFunction, optimizer, trainLoader):
    for inputs, labels in trainLoader:
        predictions = model(inputs)
        loss = lossFunction(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
def trainModel(model, lossFunction, optimizer, trainLoader, testLoader, numEpochs=2):
    for epoch in range(numEpochs):
        print("Training epoch", epoch + 1)
        trainIteration(model, lossFunction, optimizer, trainLoader)
        evaluate(model, testLoader, epoch=epoch)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    trainingSet = datasets.MNIST('data', download=True, transform=transform, train=True)
    testSet = datasets.MNIST('data', download=True, transform=transform, train=False)
    trainLoader = torch.utils.data.DataLoader(trainingSet, batch_size=64, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True)

#    for chunk in trainLoader:
#        displayBatch(chunk)
#        break

    model = TutorialModel()
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
 
    trainModel(model, lossFunction, optimizer, trainLoader, testLoader)
 
