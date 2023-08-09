import matplotlib.pyplot as plot
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

class TutorialModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Here's our first convolution layer
        # We're going to scroll a 3x3 window across the image
        # What do the other numbers mean?
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=1, padding=1)

        # A rectified linear unit (ReLU) is an activation function
        # that introduces the property of non-linearity to a deep
        # learning model and solves the vanishing gradients issue.
        self.act1 = nn.ReLU()
        # Dropout does what you'd expect. It randomly zeros certain
        # elements in the network to avoid overfitting 
        self.drop1 = nn.Dropout(0.3)

        # Why aren't these numbers the same as the previous ones?
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        # Max Pooling takes a maximum value over an input field.  It
        # helps put images in starker relief.  Think of it like
        # ratcheting up the contrast on a TV
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Reshape the data into a 1-d tensor
        self.flat = nn.Flatten()

        # Use a linear transform to reduce the data to the target size
        self.fc3 = nn.Linear(6272, 64)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        # Why are we reducing to 10 outputs?
        self.fc4 = nn.Linear(64, 10)
        self.chain = [
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

        
    def forward(self, x):
        intermediate = x
        for fn in self.chain:
            intermediate = fn(intermediate)
        return intermediate

    
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

    model = TutorialModel()

    # We need to define a loss function to help train the model The
    # loss function models how well the current iteration of the model
    # fits to the data.
    lossFunction = nn.CrossEntropyLoss()
    # The optimizer tunes the model in each training iteration, so that eventually we
    # converge on an optimal modeling of our input data.
    # SGD expands into Stochastic Gradient Descent
    # It's a kind of hill climbing.  The stochastic part is so we
    # don't get stuck in local minima and can eventually find the
    # global optimum.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainModel(model, lossFunction, optimizer, trainLoader, testLoader)

 
