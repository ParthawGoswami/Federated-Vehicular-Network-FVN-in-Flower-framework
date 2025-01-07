# Federated-Vehicular-Network-FVN-in-Flower-framework

## Usage
1. Install the dependencies if you are training on Jupyter notebook. Otherwise all dependencies come preinstalled on the Colab, Kaggle and CloudLab. Choose any of them.

2. Let's first, install Flower. You can install flower very conveniently from pip:
```
!pip install -q "flwr[simulation]" flwr-datasets
```
3. Import NumPyClient from flower client and run FVN_setup.

4. A Car Client is a simple Python class with four distinct methods:

fit(): With this method, the client does on-device training for a number of epochs using its own data. 

evaluate(): With this method, the server can evaluate the performance of the global model on the local validation set of a client. This can be used for instance when there is no centralised dataset on the server for test. 

Run Client setup and client_fn will return a CarClient that uses a specific data partition (partition-id). The index of the partition is set internally during the simulation.

5. FedAvg derives a new version of the global model by taking the average of all the models sent by clients participating in the round.  

start the simulation and check how the distributed accuracy goes up as training progresses while the loss goes down.

6. Plotting the results:
```
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(accuracy_per_round) + 1), accuracy_per_round, marker="o", label="Accuracy")
plt.title("Flower framework output")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.xticks(range(1, len(accuracy_per_round) + 1))
plt.grid(True)
plt.legend()
plt.show()
```
