import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def featureNormalizer(array_data):
    array_data = array_data/0.51
    return array_data

def targetsNormalizer(array_data):
    array_data = array_data/100
    return array_data

def loadSensorDataTrain(trainTest):
    start_idx = 0
    end_idx = 1600 # 2 cycles 280 around

    col_names = ["idx","pressure","sensor_integrated_voltage", "sensor_real_voltage",
                 "bending_angle","angle_extra","pressure_extra"]
    df0 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/1.txt", delimiter=",", names=col_names)
    df1 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/2.txt", delimiter=",", names=col_names) 
    df2 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/3.txt", delimiter=",", names=col_names) 
    df3 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/4.txt", delimiter=",", names=col_names) 
    df4 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/5.txt", delimiter=",", names=col_names) 
    df5 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/6.txt", delimiter=",", names=col_names)
    df6 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/7.txt", delimiter=",", names=col_names) 
    df7 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/8.txt", delimiter=",", names=col_names) 
    df8 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/9.txt", delimiter=",", names=col_names)
    df9 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/10.txt", delimiter=",", names=col_names) 
    df10 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/12.txt", delimiter=",", names=col_names) 
    df11 = pd.read_csv(f"piezoelectric_dataset/{trainTest}/15.txt", delimiter=",", names=col_names) 
    
    ### Sensor voltage 10 ------> base signal

    sensor_voltage1 = df1["sensor_real_voltage"].values[start_idx:end_idx]
    sensor_voltage3 = df3["sensor_real_voltage"].values[start_idx:end_idx]
    sensor_voltage4 = df4["sensor_real_voltage"].values[start_idx:end_idx] 
    sensor_voltage5 = df5["sensor_real_voltage"].values[start_idx:end_idx]
    sensor_voltage6 = df6["sensor_real_voltage"].values[start_idx:end_idx] 
    sensor_voltage7 = df7["sensor_real_voltage"].values[start_idx:end_idx] 
    sensor_voltage8 = df8["sensor_real_voltage"].values[start_idx:end_idx] 
    sensor_voltage9 = df9["sensor_real_voltage"].values[start_idx:end_idx] 
    sensor_voltage10 = df10["sensor_real_voltage"].values[start_idx:end_idx]
    sensor_voltage11 = df11["sensor_real_voltage"].values[start_idx:end_idx] 

    bending_angle1 = df1["bending_angle"].values[start_idx:end_idx]
    bending_angle3 = df3["bending_angle"].values[start_idx:end_idx]
    bending_angle4 = df4["bending_angle"].values[start_idx:end_idx]
    bending_angle5 = df5["bending_angle"].values[start_idx:end_idx]
    bending_angle6 = df6["bending_angle"].values[start_idx:end_idx]
    bending_angle7 = df7["bending_angle"].values[start_idx:end_idx]
    bending_angle8 = df8["bending_angle"].values[start_idx:end_idx]
    bending_angle9 = df9["bending_angle"].values[start_idx:end_idx]
    bending_angle10 = df10["bending_angle"].values[start_idx:end_idx]
    bending_angle11 = df11["bending_angle"].values[start_idx:end_idx]

    ## Filter out the extreme outliers from the signals
    new_sensor_signal6 = sensor_voltage6[bending_angle6<=100]
    new_bending_angle6 = bending_angle6[bending_angle6<=100]

    sensor_voltage1 = torch.FloatTensor(sensor_voltage1).unsqueeze(1) 
    sensor_voltage3 = torch.FloatTensor(sensor_voltage3).unsqueeze(1)
    sensor_voltage4 = torch.FloatTensor(sensor_voltage4).unsqueeze(1)
    sensor_voltage5 = torch.FloatTensor(sensor_voltage5).unsqueeze(1)
    sensor_voltage6 = torch.FloatTensor(new_sensor_signal6).unsqueeze(1)
    sensor_voltage7 = torch.FloatTensor(sensor_voltage7).unsqueeze(1)
    sensor_voltage8 = torch.FloatTensor(sensor_voltage8).unsqueeze(1)
    sensor_voltage9 = torch.FloatTensor(sensor_voltage9).unsqueeze(1)
    sensor_voltage10 = torch.FloatTensor(sensor_voltage10).unsqueeze(1)
    sensor_voltage11 = torch.FloatTensor(sensor_voltage11).unsqueeze(1)

    bending_angle1 = torch.FloatTensor(bending_angle1).unsqueeze(1)
    bending_angle3 = torch.FloatTensor(bending_angle3).unsqueeze(1)
    bending_angle4 = torch.FloatTensor(bending_angle4).unsqueeze(1)
    bending_angle5 = torch.FloatTensor(bending_angle5).unsqueeze(1)
    bending_angle6 = torch.FloatTensor(new_bending_angle6).unsqueeze(1)
    bending_angle7 = torch.FloatTensor(bending_angle7).unsqueeze(1)
    bending_angle8 = torch.FloatTensor(bending_angle8).unsqueeze(1)
    bending_angle9 = torch.FloatTensor(bending_angle9).unsqueeze(1)
    bending_angle10 = torch.FloatTensor(bending_angle10).unsqueeze(1)
    bending_angle11 = torch.FloatTensor(bending_angle11).unsqueeze(1)

    exp1 = TensorDataset(sensor_voltage1, bending_angle1)
    exp3 = TensorDataset(sensor_voltage3, bending_angle3)
    exp4 = TensorDataset(sensor_voltage4, bending_angle4)
    exp5 = TensorDataset(sensor_voltage5, bending_angle5)
    exp6 = TensorDataset(sensor_voltage6, bending_angle6)
    exp7 = TensorDataset(sensor_voltage7, bending_angle7)
    exp8 = TensorDataset(sensor_voltage8, bending_angle8)
    exp9 = TensorDataset(sensor_voltage9, bending_angle9)
    exp10 = TensorDataset(sensor_voltage10, bending_angle10)
    exp11 = TensorDataset(sensor_voltage11, bending_angle11)

    exps_combined = [exp10, exp1, exp3, exp4, exp5, 
                     exp6, exp7, exp8, exp9, exp11]
    exp_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return exp_ids, exps_combined

def trainTestExp(train=False):
    if train:
        return loadSensorDataTrain("train")
    else:
        return loadSensorDataTrain("test")


if __name__=="__main__":
    exp_ids, exp_combined = loadSensorDataTrain()
    print(exp_ids, exp_combined)