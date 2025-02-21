import flwr as fl
import numpy as np
import time
import matplotlib.pyplot as plt

# Parameters
num_clients = 100  # Total number of sampled clients
vector_sizes = [100000, 200000, 300000, 400000, 500000]  # Model vector sizes
client_dropout_rate = 0.05  # 5% dropout rate

def simulate_client_dropout(clients, dropout_rate):
    """Simulate client dropout after key-sharing stage."""
    num_dropped_clients = int(len(clients) * dropout_rate)
    active_clients = clients[:-num_dropped_clients] if num_dropped_clients > 0 else clients
    return active_clients

# Define Flower client class using Secure Aggregation (Salvia)
class FLClient(fl.client.NumPyClient):
    def __init__(self, vector_size):
        self.vector_size = vector_size
        self.local_vector = np.random.randint(0, 2**24, size=vector_size, dtype=np.int32)
    
    def get_parameters(self, config):
        return [self.local_vector]
    
    def fit(self, parameters, config):
        return self.get_parameters(config), len(self.local_vector), {}
    
    def evaluate(self, parameters, config):
        return 0.0, len(self.local_vector), {}

# Server-side secure aggregation simulation
def secure_aggregation_simulation(vector_size, dropout_rate):
    start_time = time.time()
    
    # Configure Flower with secure aggregation (Salvia)
    strategy = fl.server.strategy.FedAvg()
    
    # Simulate client dropout
    all_clients = [FLClient(vector_size) for _ in range(num_clients)]
    active_clients = simulate_client_dropout(all_clients, dropout_rate)
    
    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: active_clients[int(cid) % len(active_clients)],
        num_clients=len(active_clients),
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )
    
    end_time = time.time()
    computation_time = end_time - start_time
    total_data_transfer = len(active_clients) * vector_size * 4  # Assuming 4 bytes per 24-bit integer
    
    return computation_time, total_data_transfer

# Run simulation and collect results
results = {}
for vector_size in vector_sizes:
    results[vector_size] = {}
    comp_time, data_transfer = secure_aggregation_simulation(vector_size, 0.0)
    results[vector_size]['without_dropout'] = (comp_time, data_transfer)
    
    comp_time_dropout, data_transfer_dropout = secure_aggregation_simulation(vector_size, client_dropout_rate)
    results[vector_size]['with_dropout'] = (comp_time_dropout, data_transfer_dropout)

# Plot results
vector_sizes_labels = ["100k entries", "200k entries", "300k entries", "400k entries","500k entries"]
comp_times = [results[v]['without_dropout'][0] for v in vector_sizes]
comp_times_dropout = [results[v]['with_dropout'][0] for v in vector_sizes]

y_pos = np.arange(len(vector_sizes))
# plt.barh(y_pos - 0.1, comp_times, 0.2, label="No Dropout")
plt.barh(y_pos, comp_times_dropout, 0.4, label="5% Dropout")
plt.yticks(y_pos, vector_sizes_labels)
plt.xlabel("Computation Time (seconds)")
plt.title("Secure Aggregation Computation Time vs. Vector Size (Flower)")
plt.legend()
plt.show()
