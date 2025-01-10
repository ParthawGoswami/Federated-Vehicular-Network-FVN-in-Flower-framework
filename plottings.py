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
