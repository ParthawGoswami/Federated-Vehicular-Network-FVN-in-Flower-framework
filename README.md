# Federated-Vehicular-Network-FVN-in-Flower-framework

## List of Dependencies

* Python 3.10: Install with
  ```bash
  sudo apt install python3.10
* Pandas 2.1.0: Install with
  ```bash
  pip install pandas==2.1.0
* Matplotlib 3.8.0: Install with pip install matplotlib==3.8.0
* PyTorch 2.0.1+cpu: Install with pip install torch==2.0.1+cpu
* TorchVision 0.15.2+cpu: Install with pip install torchvision==0.15.2+cpu
* NumPy 1.24.4: Install with pip install numpy==1.24.4
* Flower Simulation 1.5.0: Install with pip install "flwr[simulation]==1.5.0"
* Flower Datasets 1.0.0: Install with pip install flwr-datasets==1.0.0
* JAX 0.4.14: Install with pip install jax==0.4.14
* JAXlib 0.4.14+cpu: Install with pip install jaxlib==0.4.14+cpu

## Usage
1. Install the dependencies if you are training on Jupyter notebook. Otherwise all dependencies come preinstalled on the Colab, Kaggle and CloudLab. Choose any of them.

2. Let's first, install Flower. You can install flower very conveniently from pip:
```
!pip install -q "flwr[simulation]" flwr-datasets
```
3. Import NumPyClient from flower client and run FVN_setup.

4. A Car Client is a simple Python class with four distinct methods:

* fit(): With this method, the client does on-device training for a number of epochs using its own data. 
* evaluate(): With this method, the server can evaluate the performance of the global model on the local validation set of a client. This can be used for instance when there is no centralised dataset on the server for test.
* get_parameters(): Retrieves the model's parameters and converts them to NumPy arrays for transmission to the server.
* set_parameters(): Converts received parameters (in NumPy format) back to PyTorch tensors and updates the model's state dictionary.

Run Client setup and client_fn will return a CarClient that uses a specific data partition (partition-id). The index of the partition is set internally during the simulation.

5. FedAvg derives a new version of the global model by taking the average of all the models sent by clients participating in the round. Start the simulation and check how the distributed accuracy goes up as training progresses while the loss goes down.

6. Plotting the results:
```
from typing import List
from flwr.common import Metrics
accuracy_per_round = []
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy_per_round.append(sum(accuracies) / sum(examples))
    return {"accuracy": sum(accuracies) / sum(examples)}

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
