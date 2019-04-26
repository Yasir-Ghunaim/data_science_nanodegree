import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn, optim
from collections import OrderedDict
from utility import *

def create_model(cat_to_name, device, args):
    if args["arch"] == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)
    
    # Freeze parameters 
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier 
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, args["hidden_units"])),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(args["hidden_units"], len(cat_to_name))),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args["learning_rate"]))
    model.to(device)
    
    return model, optimizer, criterion


def train_model(model, optimizer, criterion, trainloader, validloader, device, epochs, print_every):
  print("Start training... ")
  for epoch in range(epochs):
      running_loss = 0
      steps = 0
      for inputs, labels in trainloader:
          steps += 1
          # Move input and label tensors to the default device
          inputs, labels = inputs.to(device), labels.to(device)
          
          optimizer.zero_grad()
          
          logps = model.forward(inputs)
          loss = criterion(logps, labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()
          
          # Validation pass accuracy
          if steps % print_every == 0:
              validation_loss = 0
              accuracy = 0
              model.eval()
              with torch.no_grad():
                  for inputs, labels in validloader:
                      inputs, labels = inputs.to(device), labels.to(device)
                      logps = model.forward(inputs)
                      batch_loss = criterion(logps, labels)
                      
                      validation_loss += batch_loss.item()
                      
                      # Calculate accuracy
                      ps = torch.exp(logps)
                      top_p, top_class = ps.topk(1, dim=1)
                      equals = top_class == labels.view(*top_class.shape)
                      accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                      
              print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Step {steps}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
              running_loss = 0
              model.train()
  return model

def test_model(model, criterion, testloader, device):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")
    model.train()
    
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    torchImage = torch.from_numpy(image)
    torchImage = torchImage[None, : ,:, :]
    
    with torch.no_grad():
        model = model.double()
        model = model.to(device)
        torchImage = torchImage.to(device)
        logps = model.forward(torchImage)
                    
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    #Push results to cpu and convert to numpy
    top_p = top_p.to("cpu")
    top_class = top_class.to("cpu")
    top_p = top_p[0].numpy()
    top_class = top_class[0].numpy()
    
    classNumbers = []
    for targetIndex in top_class:
        # Convert from index to class number 
        for classNumber, index in model.class_to_idx.items():
            if index == targetIndex:
                classNumbers.append(str(classNumber))
    
    return top_p, classNumbers