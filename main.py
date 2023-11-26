
import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

from common import Config
from train import train
import test
import _utils

import seaborn as sns
import matplotlib.pyplot as plt
    

def _seed_everything(seed):
    """
    The function `_seed_everything` sets the seed for various random number generators in order to
    ensure reproducibility in a Python program.
    
    :param seed: The "seed" parameter is an integer value that is used to initialize the random number
    generators in various libraries such as random, numpy, and torch. By setting a specific seed value,
    you can ensure that the random numbers generated during the execution of your code are reproducible.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def main():
    """
    The main function trains a model, tests it, and evaluates its performance using various metrics.
    """
 
    num_replicates = Config.num_replicates
    num_classes = Config.num_classes
    epochs = Config.epochs
    seed = Config.seed
    
    # This code block initializes empty arrays to store the evaluation metrics for each replicate. The
    # arrays are initialized with zeros and have dimensions based on the number of replicates
    # (`num_replicates`) and the number of classes (`num_classes`).
    if Config.result_out_dir is not None:
        conf_mats = np.zeros((num_replicates, num_classes, num_classes))
        p_scores = np.zeros((num_replicates, num_classes))
        r_scores = np.zeros((num_replicates, num_classes))
        f1_scores = np.zeros((num_replicates, num_classes))
        k_scores = np.zeros((num_replicates,))
        oa_arr = np.zeros((num_replicates,))
    
    # repeat this code for each replicate
    for rep in range(num_replicates):
        # fix random seeds
        _seed = seed + rep
        _seed_everything(_seed)
        
        # prepare data
        X, y, X_test, y_test = _utils._get_dataset(Config.dataset, Config.data_dir)
        train_set, val_set = _utils._split_train_val(X, y, train_ratio=1.0)
        train_loader = DataLoader(dataset=train_set, batch_size=Config.batch_size, shuffle=True)
        
        # prepare model and optimizer
        class_weights = _utils._get_class_weights(y, num_classes, Config.mask_undefined)
        if Config.use_gpu:
            class_weights = class_weights.cuda()
        if Config.mask_undefined:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            
        # This code block is checking if the `Config.ckpt_dir` variable is not empty. If it is not
        # empty, it loads a checkpoint file using `torch.load()` function. The checkpoint file is
        # named `'ckpt_rep%d_epoch100.pth'` where `%d` is replaced with the value of `rep`. If
        # `Config.ckpt_dir` is empty, it sets `ckpt` to `None`.
        if Config.ckpt_dir:
            ckpt = torch.load(os.path.join(Config.ckpt_dir, 'ckpt_rep%d_epoch100.pth'%rep))
        else:
            ckpt = None
            
        # This code block is selecting the model architecture based on the value of `Config.model`.
        if Config.model == 'tb_cnn':
            model = _utils._get_model(Config.model, ckpt=ckpt, in_channel_branch=[64, 2], n_classes=num_classes, patch_size=Config.sample_radius)
        else:
            # ResNets
            model = _utils._get_model(Config.model, ckpt=ckpt, input_channels=X_test.shape[0], n_classes=num_classes, 
                               use_dgconv=Config.use_dgconv, use_init=Config.use_init, fix_groups=Config.fix_groups)
        if Config.use_gpu:
            model = model.cuda()
            
        optimizer = _utils._get_optimizer(model, opt_name=Config.optimizer, lr=Config.lr, ckpt=ckpt)
        
        # train
        model, losses = train(model, train_loader, optimizer, criterion, 
                              num_epochs=epochs, rep=rep, mask_undefined=Config.mask_undefined, 
                              save_ckpt_dir=Config.save_ckpt_dir, use_gpu=Config.use_gpu, 
                              lr_schedule=Config.lr_schedule, verbose=True)
        # test and eval
        if Config.result_out_dir is not None:
            pred_map = test.test_clf(model, X_test, sample_radius=Config.sample_radius)
            print(pred_map)
            
            y_pred_all = pred_map[y_test>0]
            y_true_all = y_test[y_test>0]
            conf_mats[rep] = confusion_matrix(y_true_all, y_pred_all)
            p_scores[rep] = precision_score(y_true_all, y_pred_all, average=None)
            r_scores[rep] = recall_score(y_true_all, y_pred_all, average=None)
            f1_scores[rep] = f1_score(y_true_all, y_pred_all, average=None)
            k_scores[rep] = cohen_kappa_score(y_true_all, y_pred_all)
            oa_arr[rep] = np.sum(conf_mats[rep]*np.eye(num_classes, num_classes)) / np.sum(conf_mats[rep])
                
    # print information
    print("Conf mats : ", conf_mats)
    print('losses : ', losses)
    print('p_scores , ', p_scores)
    print('r_scores , ', r_scores)
    print('f1_scores: ', f1_scores)
    print('k_scores : ', k_scores)
    print('oa_arr ', oa_arr)
    
    # print confusion matrix
    sns.heatmap(conf_mats[0], annot=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # print result map
    plt.imshow(pred_map, cmap='viridis')
    plt.colorbar()
    plt.show()
        
   
if __name__ == '__main__':
    main()

