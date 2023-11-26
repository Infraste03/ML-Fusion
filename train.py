
# Model training


import os
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR

def _checkpoint(model, optimizer, ckpt_path, loss=0.0):
    """
    The function `_checkpoint` saves the state of a model, optimizer, and loss to a checkpoint file.
    
    :param model: The `model` parameter is the neural network model that you want to save. It should be
    an instance of a PyTorch `nn.Module` subclass
    :param optimizer: The optimizer is an object that implements the optimization algorithm. It is
    responsible for updating the model's parameters based on the computed gradients during the training
    process. Examples of optimizers in PyTorch include SGD (Stochastic Gradient Descent), Adam, and
    RMSprop
    :param ckpt_path: The `ckpt_path` parameter is the path where you want to save the checkpoint file.
    It should be a string that specifies the file path, including the file name and extension. For
    example, if you want to save the checkpoint file as "checkpoint.pth", you can set `ckpt_path` as
    :param loss: The `loss` parameter is the value of the loss function at the current checkpoint. It is
    used to save the current loss value along with the model and optimizer states
    """
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, ckpt_path)
    


    """
    The `train` function trains a given model using a specified optimizer and criterion for a specified
    number of epochs, with optional features such as learning rate scheduling, saving checkpoints, and
    printing verbose output.
    
    :param model: The model is the neural network model that you want to train. It should be an instance
    of a PyTorch model class
    :param train_loader: The train_loader parameter is a DataLoader object that provides the training
    data in batches. It is used to iterate over the training dataset during training
    :param optimizer: The optimizer is an object that specifies the optimization algorithm to use during
    training. It is responsible for updating the model's parameters based on the computed gradients.
    Some commonly used optimizers include SGD (Stochastic Gradient Descent), Adam, and RMSprop
    :param criterion: The criterion is the loss function used to measure the model's performance. It is
    typically a function that takes the predicted output and the target output and computes a scalar
    value that represents the model's error. Common examples include mean squared error (MSE),
    cross-entropy loss, and binary cross-entropy
    :param num_epochs: The number of training epochs, which is the number of times the model will
    iterate over the entire training dataset, defaults to 100 (optional)
    :param rep: The `rep` parameter is an integer that represents the repetition number or iteration
    number of the training process. It is used to differentiate between different iterations of
    training, especially when saving checkpoints, defaults to 0 (optional)
    :param mask_undefined: The `mask_undefined` parameter is a boolean flag that determines whether to
    mask undefined labels during training. If set to `True`, the undefined labels (label=0) will be
    masked and not used for computing the loss and accuracy. This can be useful when dealing with
    datasets that have undefined or, defaults to True (optional)
    :param save_ckpt_dir: The `save_ckpt_dir` parameter is the directory where the checkpoints of the
    model will be saved. Checkpoints are saved periodically during training to allow for resuming
    training from a specific epoch or for evaluating the model's performance at different stages of
    training
    :param use_gpu: A boolean indicating whether to use GPU for training. If set to True, the model and
    data will be moved to the GPU for faster computation. If set to False, the training will be done on
    the CPU, defaults to True (optional)
    :param lr_schedule: The `lr_schedule` parameter is a list of milestones at which the learning rate
    should be reduced. It is used to implement a learning rate schedule during training. Each milestone
    corresponds to an epoch number, and when the current epoch number matches a milestone, the learning
    rate is multiplied by a factor of
    :param verbose: The `verbose` parameter controls whether or not to print the training progress
    during each epoch. If set to `True`, it will print the loss and accuracy for each epoch. If set to
    `False`, it will not print anything, defaults to True (optional)
    :return: the trained model and an array of losses for each epoch.
    """
def train(model, train_loader, optimizer, criterion, num_epochs=100, rep=0, mask_undefined=True, 
          save_ckpt_dir=None, use_gpu=True, lr_schedule=None, verbose=True):
    
    if lr_schedule is not None:
        scheduler = MultiStepLR(optimizer, milestones=lr_schedule, gamma=0.1)
    losses = np.zeros((num_epochs,))
    model.train()
    for epoch in range(num_epochs):
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, (x, label) in enumerate(train_loader):

            if mask_undefined:
                target = label.clone()
                target[target==0] = 1
                target -= 1
                mask = label!=0
                mask = mask.float()
                if use_gpu:
                    mask = mask.cuda()
                    target = target.cuda()
            if use_gpu:
                x = x.cuda()
                label = label.cuda()

            length = len(train_loader)
            optimizer.zero_grad()

            pred = model(x)
            if mask_undefined:
                loss = criterion(pred, target)
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if verbose:
                _, predicted = torch.max(pred.data, 1)
                if mask_undefined:
                    total += mask.sum()
                    predicted += 1
                    correct += ((predicted.eq(label.data))*mask).cpu().sum()
                else:
                    total += len(label)
                    correct += (predicted.eq(label.data)).cpu().sum() 
            
        if verbose:
            if epoch % 10 == 0:                      
                print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, sum_loss / (i + 1), 100. * correct / total))
        
        if save_ckpt_dir:
            if epoch % 100 == 99:
                ckpt_path = os.path.join(save_ckpt_dir, 'ckpt_rep%d_epoch%d.pth'%(rep,epoch))
                _checkpoint(model, optimizer, ckpt_path, sum_loss)
            
        losses[epoch] = sum_loss
        if lr_schedule is not None:
            scheduler.step()
    
    return model, losses

