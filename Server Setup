device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from flwr.simulation import start_simulation
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters

# Initialize the model
model = SimpleCNN().to(device)

parameters = [param.detach().numpy() for param in simulation.global_model.state_dict().values()]

# Define the strategy
strategy = FedAvg(
    fraction_fit=0.1,
    fraction_evaluate=0.5,
    evaluate_metrics_aggregation_fn=test_model(simulation.global_model.to(device), test_loader),
)

# Define simulation configuration
config = {
    "num_rounds": 20,  # Set the number of federated learning rounds
}

# Start the simulation
start_simulation(
    client_fn=client_fn,         # Client creation function
    num_clients=num_clients,  # Number of clients
    strategy=strategy,           # Federated strategy
    config=config                # Simulation configuration
)
