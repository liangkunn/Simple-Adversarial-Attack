import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

class GradientAttack():
        
        """
        Implement a one step gradient attack

        Step 1: Calculate the Loss and then calculate the gradient of the Loss w.r.t the image

        Step 2: Add the gradient (or its sign for each pixel), multiplied by a small step size, to the original image

        might need to clamp the modified image to make sure the values of each pixel are between [0,1]
        """
        
        def __init__(self, loss, epsilon):
            """
            
            """
            self.loss = loss
            self.epsilon = epsilon

        def forward(self, x, y_true, model):
            """
            
            """
            y_pred = model(x)
            loss_ = self.loss(y_pred, y_true)

            loss_.backward()

            x_grad_sign = torch.sign(x.grad.data)
            x_adv = x + x_grad_sign * self.epsilon
            x_adv.data.clamp_(0,1)


            return x_adv
        

class Net(nn.Module):
    def __init__(self, input_size, width, num_classes):
        super(Net, self).__init__()

        ##feedfoward layers:
        self.ff1 = nn.Linear(input_size, width) #input

        self.ff2 = nn.Linear(width, width) #hidden layers
        self.ff3 = nn.Linear(width, width)

        self.ff_out = nn.Linear(width, num_classes) #logit layer     

        ##activations:
        self.relu = nn.ReLU()
                
    def forward(self, input_data):
        out = self.relu(self.ff1(input_data)) 
        out = self.relu(self.ff2(out)) 
        out = self.relu(self.ff3(out))
        out = self.ff_out(out)
        return out #returns class probabilities for each image

                          