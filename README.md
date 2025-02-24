# Adaptive Drift Compensation for Soft Sensorized Finger Using Continual Learning
Strain sensors are gaining popularity in soft robotics for acquiring tactile data due to their flexibility and
ease of integration. Tactile sensing plays a critical role in soft grippers, enabling them to safely interact with unstructured
environments and precisely detect object properties. However, a significant challenge with these systems is their high nonlinearity, time-varying behavior, and long-term signal drift. In
this paper, we introduce a continual learning (CL) approach to model a soft finger equipped with piezoelectric-based strain
sensors for proprioception. To tackle the aforementioned challenges, we propose an adaptive CL algorithm that integrates a Long Short-Term Memory (LSTM) network with a memory buffer for rehearsal and includes a regularization term to keep the model’s decision boundary close to the base signal while adapting to time-varying drift. We conduct nine different experiments, resetting the entire setup each time to demonstrate
signal drift. We also benchmark our algorithm against two other methods and conduct an ablation study to assess the impact of different components on the overall performance.


<img src="https://github.com/nilay121/SensorDriftCompensationUsingCL/blob/main/archi_finger.png" height="400px" width="1000px">architecture2.png)
