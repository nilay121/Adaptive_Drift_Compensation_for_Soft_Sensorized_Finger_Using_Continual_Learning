# Adaptive Drift Compensation for Soft Sensorized Finger Using Continual Learning
## Abstract
Strain sensors are gaining popularity in soft robotics for acquiring tactile data due to their flexibility and
ease of integration. Tactile sensing plays a critical role in soft grippers, enabling them to safely interact with unstructured
environments and precisely detect object properties. However, a significant challenge with these systems is their high nonlinearity, time-varying behavior, and long-term signal drift. In
this paper, we introduce a continual learning (CL) approach to model a soft finger equipped with piezoelectric-based strain
sensors for proprioception. To tackle the aforementioned challenges, we propose an adaptive CL algorithm that integrates a Long Short-Term Memory (LSTM) network with a memory buffer for rehearsal and includes a regularization term to keep the model’s decision boundary close to the base signal while adapting to time-varying drift. We conduct nine different experiments, resetting the entire setup each time to demonstrate
signal drift. We also benchmark our algorithm against two other methods and conduct an ablation study to assess the impact of different components on the overall performance.

## Model Architecture and the Soft Sensorized Finger
<img src="https://github.com/nilay121/SensorDriftCompensationUsingCL/blob/main/architecture.png" height="300px" width="1000px">


The network is structured into two distinct parts. The first part, referred to as the static part, consists of a single LSTM layer with 16 nodes, followed by a flattening layer and a
fully connected layer with 64 output neurons. The second part, called the dynamic part, comprises two fully connected layers with 64 and 256 neurons, respectively, and a single
output neuron. ReLU activation is applied after each linear layer, except for the final output layer. More details can be found in the paper.

## To Run the algorithm
- Create a virtual environment, you can use the requirements.txt from our SynapNet repo
  
- Initialize you data in a proper manner according to the script "SensorDataGenerator.py"

- Then just run the main script
  ```bash
  python3 main_withAutomaticTest.py
  ```
- Kindly cite the paper, if the repository helped you in your project

## To cite the paper
  ```bash

  ```
