import matplotlib.pyplot as plot
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

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
    for epoch in range(1,numEpochs+1):
        print("Training epoch", epoch)
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

## You can uncomment the following section to get a sense for the data
## being used by pytorch for the mnist dataset.  It's worth noting
## that it uses data at a very different resolution to demo they're
## machine vision tools.  The tradeoff is that the data isn't packaged
## with the library; it has to be downloaded separately.
#    for chunk in trainLoader:
#        displayBatch(chunk)
#        break

    # This Model is equivalent to the one we produced by hand in the
    # first part.  Turns out if you're building a forward-feeding
    # network of a set of transforms, pytorch has built in
    # functionality supporting you.
    model2 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Dropout(0.3),
 
        nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2)),
 
        nn.Flatten(),
 
        nn.Linear(6272, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
 
        nn.Linear(64, 10)

        )
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
    trainModel(model2, lossFunction, optimizer, trainLoader, testLoader)
 
