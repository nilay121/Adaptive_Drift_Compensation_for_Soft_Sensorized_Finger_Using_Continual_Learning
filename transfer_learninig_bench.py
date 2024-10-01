import torch
import copy
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from utils import initialize_weights_to_ones, save_plot,\
                  buffer_datasetGenerator, ResettableDataLoader,\
                  lwf_penalty

def freezeBaseLayers(cl_model):
    ## freeze the static layer1 and layer2
    print("Freezing the static layer!!")
    for params in cl_model.baseLayer1.parameters():
        params.requires_grad = False
    for params in cl_model.baseLayer2.parameters():
        params.requires_grad = False

def unfreezeBaseLayers(cl_model):
    ## freeze the static layer1 and layer2
    print("Unfreezing the static layer!!")
    for params in cl_model.baseLayer1.parameters():
        params.requires_grad = True
    for params in cl_model.baseLayer2.parameters():
        params.requires_grad = True

def freezeAdaptiveLayer(cl_model):
    print("Freezing the adaptive layer!!")
    cl_model.adaptiveLayer.apply(initialize_weights_to_ones)
    ## Freeze the adaptive model 
    for params in cl_model.adaptiveLayer.parameters():
        params.requires_grad = False

def unfreezeAdaptiveLayer(cl_model):
    print("Unfreezing the adaptive layer!!")
    for params in cl_model.adaptiveLayer.parameters():
        params.requires_grad = True

## Test the algorithm by just activating the base model
def seqTrain(dataset, cl_model, uniform_buffer, exp_id, epochs_base, 
             epochs_adaptive, lr_adaptive, lr_base, lwf_alpha, lambda_input, 
             lambda_buffer, patience_scheduler, saved_prev_model, 
             batch_size_train, device):
       
    ## Normal training of the static initial part of the network
    criterion = nn.MSELoss()
    loss_collector = []
    dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=False)
    optimizer_base = optim.Adam(filter(lambda p: p.requires_grad, cl_model.parameters()), lr=lr_base)
    optimizer_adaptive = optim.Adam(filter(lambda p: p.requires_grad, cl_model.parameters()), 
                                    lr=lr_adaptive, weight_decay=1e-4)
    scheduler_adaptive = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_adaptive, 
                                                                    patience=patience_scheduler, 
                                                                    verbose=True)
    if exp_id == 0:
        ## freezing the adaptive layer
        freezeAdaptiveLayer(cl_model)
        for epoch in range(epochs_base):
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer_base.zero_grad()
                outputs = cl_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_base.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs_base}], Loss: {loss.item():.4f}')
            loss_collector.append(loss.item())
        ## unfreezing the adaptive layer
        unfreezeAdaptiveLayer(cl_model)
        ## freeze base layer
        freezeBaseLayers(cl_model)
    else:
        print("----------------- Combined Loss with buffer -----------------")
        for epoch in range(epochs_adaptive):
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer_adaptive.zero_grad()
                outputs = cl_model(inputs)
                loss = lambda_input*criterion(targets, outputs) 
                loss.backward()
                optimizer_adaptive.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs_adaptive}], Loss: {loss.item():.4f}')
            scheduler_adaptive.step(loss)
            loss_collector.append(loss.item())

    # ## Store one cycle for each task in buffer
    # uniform_buffer[exp_id] = [dataset.tensors[0].detach().numpy()[0:200],
    #                         dataset.tensors[1].detach().numpy()[0:200]]
    # ## update the previous model with the current model
    # saved_prev_model = copy.deepcopy(cl_model)
    return cl_model, saved_prev_model, loss_collector

def seqTest(test_dataset, exp_id_train, exp_id_test, model, device):
    batch_size = test_dataset.__len__()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    for features, targets in test_loader:
        features = features.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            predictions = model(features).squeeze()
        original_pred = targets.squeeze()
    original_pred = original_pred.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    rmse_error = root_mean_squared_error(original_pred, predictions)
    #sent again to training phase
    model.train()
    print("RMSE error for exp id train test ",exp_id_train,exp_id_test, 
          "is", rmse_error, " rad")
    # save the image for predictions
    save_plot(original_pred, predictions, f"{exp_id_train}{exp_id_test}")
    
    return rmse_error, original_pred, predictions


