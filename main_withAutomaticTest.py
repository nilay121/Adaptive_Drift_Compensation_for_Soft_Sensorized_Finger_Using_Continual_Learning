import torch
import numpy as np
from cL_model import hybridModel
from cL_algorithm import seqTrain, seqTest
from scipy.stats import wasserstein_distance
from utils import save_dictionary, save_array
from SensorDataGenerator import trainTestExp


def main():

    trial_overall_error = []
    experiment_loss = {}
    model_predictions = {}
    ground_truth = {}
    num_trails = 5
    n_tasks = 10
    
    for counter in range(num_trails):
        ##Training Paramaeters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ## model epochs
        epochs_base = 550  #450
        epochs_adaptive = 850
        batch_size_train = 128
        ## learning rates
        lr_base = 1e-3
        lr_adaptive = 1e-3
        ## lwf alpha
        lwf_alpha = 0.95
        ## loss weightage
        lambda_input = 1
        lambda_buffer = 2
        ## Other params
        patience_scheduler = 1000
        saved_prev_model = None
        was_threshold = 0.0005 ## calculated analytically

        exp_ids_train, exps_train = trainTestExp(train=True)
        exp_ids_test, exps_test = trainTestExp(train=False)

        ## Initialize model
        adaptive_model = hybridModel(1, 1).to(device)
        ## Initialize a buffer for rehearsal
        uniform_buffer = {}

        for exp_id_train, experience_train in zip(exp_ids_train, exps_train):
            test_error = []
            ## Store the first base hysteresis signal for comparison against the new incoming ones
            if exp_id_train == 0:
                ##initialize array with 0s
                was_dist_temp = np.zeros(10)
            else:
                was_dist_temp = []
                ## perform wasserstein distance test on the incoming signal
                new_incoming_signal = experience_train.tensors[0][0:200].reshape(-1)
                for i in range(len(uniform_buffer.keys())):
                    was_dist = wasserstein_distance(np.array(list(uniform_buffer.values())[i][0]).reshape(-1), 
                                                    new_incoming_signal)
                    was_dist_temp.append(was_dist)
                was_dist_temp = np.array(was_dist_temp)
            ## Condition to check if the new signal has significant differences in comparison to the 
            ## base signal
            was_dist_check = (was_dist_temp <= was_threshold)
            if was_dist_check.sum() >= 1 and exp_id_train>0:
                print("*"*10, "Incoming signal similar to the base signal", "*"*10) 
                print("*"*10, "Skipping training on the new signal", "*"*10)
            else:
                print(f"-----------------------> Train Exp Id {exp_id_train} <-----------------------")
                model, saved_prev_model, temp_loss = seqTrain(experience_train, adaptive_model, uniform_buffer, 
                                exp_id_train, epochs_base, epochs_adaptive, lr_adaptive, lr_base, lwf_alpha,
                                lambda_input, lambda_buffer, patience_scheduler, 
                                saved_prev_model, batch_size_train, device)

            for exp_id_test, experience_test in zip(exp_ids_test, exps_test):
                print(f"-----------------------> Test Exp Id {exp_id_test} <-----------------------")
                rmse_error, org_gt, model_pred = seqTest(experience_test, exp_id_train, exp_id_test, model, device)
                test_error.append(rmse_error)
                ## Append the model predictions and the ground truth in the dictionary with the current run as the key
                ## Store the predictions after the model has been trained on all the experiences (experiments)
                if exp_id_train==(n_tasks-1):
                    if exp_id_test==0:
                        model_predictions[counter] = [model_pred]
                        ground_truth[counter] = [org_gt]
                    else:
                        model_predictions[counter].append(model_pred)
                        ground_truth[counter].append(org_gt)

            ## append the loss for current task in the dictionary
            if exp_id_train==0:
                experiment_loss[counter] = [temp_loss]
            else:
                experiment_loss[counter].append(temp_loss)

        trial_overall_error.append(np.array(test_error).reshape(-1,1))
    trial_overall_error = np.concatenate(trial_overall_error, axis=1)

    ## save the loss dictionary, predictions dictionary, ground truth dictionary
    save_dictionary(experiment_loss, f"loss_each_task_count{num_trails}_withAdaptive50_withReplay")
    save_dictionary(model_predictions, f"model_predictions_count{num_trails}_withAdaptive50_withReplay")
    save_dictionary(ground_truth, f"ground_truth_count{num_trails}_withAdaptive50_withReplay")
    save_array(trial_overall_error)
    ## average error for all the trials
    average_trial_error = np.average(trial_overall_error, axis=1)
    print(average_trial_error)




if __name__=="__main__":
    main()



