
# Some helper functions

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

    
def _get_dataset(ds_name, ds_dir):
    """
    The function `_get_dataset` loads a dataset based on the dataset name and directory provided.
    
    :param ds_name: The `ds_name` parameter is a string that represents the name of the dataset. It is
    used to determine which dataset implementation to use
    :param ds_dir: The `ds_dir` parameter is the directory where the dataset is located. It is a string
    that specifies the path to the directory
    :return: the result of calling the `load_dataset` function with the `ds_dir` argument.
    """
    if ds_name == 'muufl':
        from data.dataset_muufl import load_dataset
    else:
        raise NotImplementedError('Data set not implemented!')
    return load_dataset(ds_dir)
    
def _split_train_val(X, y, train_ratio=1.0):
    """
    The function `_split_train_val` splits the input data `X` and `y` into a training set and a
    validation set, with the option to specify the ratio of data to be used for training.
    
    :param X: X is a numpy array or a list containing the input features of the dataset. Each row of X
    represents a single data point, and each column represents a different feature
    :param y: The parameter `y` represents the target variable or the labels for the dataset `X`. It is
    a list or array containing the target values corresponding to each sample in `X`
    :param train_ratio: The train_ratio parameter determines the ratio of data that will be used for
    training. For example, if train_ratio is set to 0.8, 80% of the data will be used for training and
    the remaining 20% will be used for validation
    :return: two variables: `train_set` and `val_set`.
    """
    X_tensor = torch.Tensor(X)
    y_tensor = torch.LongTensor(y)
    data_set = TensorDataset(X_tensor, y_tensor)
    total_size = len(data_set)
    train_size = int(total_size*train_ratio)
    val_size = total_size - train_size
    if val_size > 0:
        train_set, val_set = torch.utils.data.random_split(data_set, [train_size, val_size])
    else:
        train_set = data_set
        val_set = None
    return train_set, val_set

def _get_class_weights(y, num_classes, mask):
    """
    The function calculates class weights based on the number of samples in each class.
    
    :param y: The parameter "y" represents the target variable or the labels of the dataset. It is a
    numpy array containing the class labels for each sample in the dataset
    :param num_classes: The number of classes in the dataset
    :param mask: The `mask` parameter is a boolean value that indicates whether to consider a mask when
    calculating the class weights. If `mask` is `True`, the function will calculate the class weights
    based on the number of samples for each class that are not masked. If `mask` is `False`, the
    :return: a torch.FloatTensor object containing the class weights.
    """
    num_samples = np.zeros((num_classes,))
    if mask:
        for i in range(1, num_classes+1):
            num_samples[i-1] = np.sum(y==i)
    else:
        for i in range(num_classes):
            num_samples[i] = np.sum(y==i)
    class_weights = [1 - (n / sum(num_samples)) for n in num_samples]
    return torch.FloatTensor(class_weights)

def _get_model(model_name='resnet18', ckpt=None, **kwargs):
    """
    The function `_get_model` returns a specified model architecture with optional pre-trained weights.
    
    :param model_name: The `model_name` parameter is used to specify the name of the model architecture
    to be used. The available options are 'resnet18', 'resnet50', and 'tb_cnn', defaults to resnet18
    (optional)
    :param ckpt: The `ckpt` parameter is used to load a pre-trained model checkpoint. It is a dictionary
    that contains the state dictionary of the model. By passing a `ckpt` value, the function will load
    the model's state dictionary from the checkpoint and assign it to the `model` object before
    returning it
    :return: a model object based on the specified model name. If a checkpoint is provided, the function
    also loads the model's state dictionary from the checkpoint.
    """
    if model_name == 'resnet18':
        from model.resnet import resnet18
        model = resnet18(**kwargs)
    elif model_name == 'resnet50':
        from model.resnet import resnet50
        model = resnet50(**kwargs)
    elif model_name == 'tb_cnn':
        from model.baseline.tb_cnn import TB_CNN
        model = TB_CNN(**kwargs)
    else:
        raise NotImplementedError('Model not implemented!')
    if ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    return model
        
def _get_optimizer(model, opt_name='adam', lr=0.001, ckpt=None):
    """
    The function `_get_optimizer` returns an optimizer object based on the specified optimizer name,
    learning rate, and optional checkpoint.
    
    :param model: The `model` parameter is the neural network model for which you want to create an
    optimizer. It should be an instance of a PyTorch `nn.Module` subclass
    :param opt_name: The `opt_name` parameter is used to specify the optimizer to be used. It can take
    two values: 'adam' or 'sgd', defaults to adam (optional)
    :param lr: The "lr" parameter stands for learning rate. It determines the step size at each
    iteration while optimizing the model. A higher learning rate can lead to faster convergence, but it
    may also cause the optimization process to overshoot the optimal solution. On the other hand, a
    lower learning rate may result in
    :param ckpt: The `ckpt` parameter is used to load a checkpoint of the optimizer's state dictionary.
    This can be useful when you want to resume training from a previously saved checkpoint. The state
    dictionary contains information about the optimizer's internal state, such as the current learning
    rate, momentum, etc. By loading the
    :return: an optimizer object.
    """
    if opt_name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer not implemented!')
    if ckpt: 
        optim.load_state_dict(ckpt['optimizer_state_dict'])
    return optim

    