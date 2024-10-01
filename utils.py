import torch
import copy
import pickle
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, ttest_ind
from torch.utils.data import DataLoader, TensorDataset

## ------------------> Hypothesis tests
def KS_test(base_distribution, new_distribution):
    ks_stat, ks_p_value = ks_2samp(base_distribution, new_distribution)
    return ks_p_value, ks_stat

def t_test(base_distribution, new_distribution):
    t_stat, t_p_value = ttest_ind(base_distribution, new_distribution)
    return t_p_value, t_stat

## ------------------> adaptive layer weight initialization
def initialize_weights_to_ones(layer):
    if isinstance(layer, nn.Linear):
        with torch.no_grad():
            ## set the weights to 1
            layer.weight.fill_(1.0)
            if layer.bias is not None:
                ## set the bias to 0
                layer.bias.fill_(0.0)

def buffer_datasetGenerator(buffer_dict):
    features = np.concatenate([buffer_dict[ids][0] for ids in [x for x in buffer_dict.keys()]])
    targets = np.concatenate([buffer_dict[ids][1] for ids in [x for x in buffer_dict.keys()]])
    ## Unsquezze an extra dimension
    features = torch.FloatTensor(features)
    targets = torch.FloatTensor(targets)
    ## Create torch dataset
    buffer_dataset = TensorDataset(features, targets)
    return buffer_dataset

## ------------------> Lwf regularization for regression
def model_discrepancy_loss(current_out, prev_out):
    # prev_model_loss = torch.nn.functional.mse_loss(prev_out, current_out)
    prev_model_loss = torch.nn.functional.l1_loss(prev_out, current_out)

    return prev_model_loss

def lwf_penalty(prev_model, current_pred, current_input):

    if prev_model is None:
        return 0
    else:
        with torch.no_grad():
            y_prev = prev_model(current_input)
            y_curr = current_pred
    yp = y_prev
    yc = y_curr
    
    dist_loss = model_discrepancy_loss(yc, yp)
    return dist_loss  

## ------------------> Reset Dataloader
## To reset the dataloader and start again from the first batch, 
## once the last batch of the dataset is reached
class ResettableDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Return the next batch
            return next(self.iterator)
        except StopIteration:
            # Reset the iterator when StopIteration is executed
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

def save_plot(org_pred, model_pred, img_name):
    plt.figure(figsize=(10, 5))
    plt.plot(org_pred, label='True Data')
    plt.plot(model_pred, label='Predicted Data')
    plt.title('Sensor signal prediction')
    plt.xlabel('Time')
    plt.ylabel('Sensor Output')
    plt.legend()
    plt.show()
    plt.savefig(f"images/testPlot{img_name}.png")
    plt.close()

def save_dictionary(data, file_name):
    with open(f'saved_metrics/{file_name}.pkl', 'wb') as fp:
        pickle.dump(data, fp)
        print('dictionary saved successfully to file')

def load_dictionary(file_path):
    with open(file_path, 'rb') as fp:
        interpolation_steps = pickle.load(fp)
    return interpolation_steps

def save_array(data):
    np.save("saved_metrics/rmse_errors.npy",data)