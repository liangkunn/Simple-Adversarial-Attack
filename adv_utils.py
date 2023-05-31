import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16, resnet50
gpu_boole = torch.cuda.is_available()

## Evaluation Functions (E.g Loss, Accuracy)
## put everything together
def train_eval(net, train_loader, loss_metric, verbose = 1, color_channel = 1):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in train_loader:

        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()

        if color_channel == 1:
            images = images.view(-1, 28*28)
        else:
            images = images

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

        loss_sum += loss_metric(outputs,labels).item()
        
    if verbose:
        print('Train accuracy: %f %%' % (100.0 * correct / total))
        print('Train loss: %f' % (loss_sum / total))

    return 100.0 * correct / total, loss_sum / total
    
def test_eval(net, test_loader, loss_metric, verbose = 1, color_channel = 1):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in test_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()

        if color_channel == 1:
            images = images.view(-1, 28*28)
        else:
            images = images       

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

        loss_sum += loss_metric(outputs,labels).item()

    if verbose:
        print('Test accuracy: %f %%' % (100.0 * correct / total))
        print('Test loss: %f' % (loss_sum / total))

    return 100.0 * correct / total, loss_sum / total

def test_eval_adv(net, test_loader, loss_metric, adv_attack, verbose = 1, color_channel = 1):
    correct = 0
    total = 0
    loss_sum = 0
    for images, labels in test_loader:
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()

        if color_channel == 1:
            images = images.view(-1, 28*28)
        else:
            images = images

        images = Variable(images, requires_grad=True)
        images = adv_attack.forward(images, Variable(labels), net)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

        loss_sum += loss_metric(outputs,labels).item()

    if verbose:
        print('Test accuracy adversarial: %f %%' % (100.0 * correct / total))
        print('Test loss adversarial: %f' % (loss_sum / total))

    return 100.0 * correct / total, loss_sum / total



#training loop:
def training_loop(epochs, train_loader, test_loader, net, loss_metric, optimizer, adv_attack, color_channel):

    log_dict = {
    'train_acc': [],
    'train_loss': [],
    'test_acc': [],
    'test_loss': [],
    'test_acc_adv': [],
    'test_loss_adv': []
    }

    print(" =======> Starting Training")

    for epoch in range(epochs):
        
        time1 = time.time() #timekeeping
        
        # the following code should not be ommitted 
        # for the purpose of upgrading model parameters.
        for i, (x,y) in enumerate(train_loader):

            if gpu_boole:
                x = x.cuda()
                y = y.cuda()

            if color_channel == 1:
                x = x.view(x.shape[0],-1)
            else:
                x = x
            #loss calculation and gradient update:

            if i > 0 or epoch > 0:
                optimizer.zero_grad()
            outputs = net.forward(x)
            loss = loss_metric(outputs,y)
            loss.backward()
            
            ##perform update:
            optimizer.step()

        # if epoch+1 % 2 == 0:
        print()
        print('============================================')
        print("Epoch", epoch+1,':')

        time2 = time.time() #timekeeping
        print('Elapsed time for epoch:',time2 - time1,'s')
        print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')

        # obtain accuracies and loss
        train_perc, train_loss = train_eval(net, train_loader, loss_metric, 1, color_channel)
        test_perc, test_loss = test_eval(net, test_loader, loss_metric, 1, color_channel)
        test_acc_adv_, test_loss_adv_ = test_eval_adv(net, test_loader, loss_metric, adv_attack, 1, color_channel)

        # append to the log_dict
        log_dict['train_acc'].append(train_perc)
        log_dict['train_loss'].append(train_loss)
        log_dict['test_acc'].append(test_perc)
        log_dict['test_loss'].append(test_loss)
        log_dict['test_acc_adv'].append(test_acc_adv_)
        log_dict['test_loss_adv'].append(test_loss_adv_)
    

    return log_dict

def initialize_model(use_resnet=True, pretrained=False, nclasses=10):
    """
    
    """
    ## Initialize Model
    if use_resnet:
        model = resnet50(pretrained=pretrained)
    else:
        model = vgg16(pretrained=True)
    ## Freeze Early Layers if Pretrained
    if pretrained:
        for parameter in model.parameters():
            parameter.requires_grad = False
    ## Update Output Layer
    if use_resnet:
        model.fc = nn.Linear(2048, nclasses)
    else:
        model.classifier._modules['6'] = nn.Linear(4096, nclasses)
    return model