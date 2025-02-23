!pip install flwr
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from flwr.client import NumPyClient
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CarClient(NumPyClient):
    def __init__(self, car, dataloader):
        super().__init__()
        self.car = car
        self.dataloader = dataloader
        self.model = SimpleCNN().to(device)

    def get_parameters(self, config):
        """Return the model parameters."""
        return [param.detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters."""
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model using the car's local data."""
        self.set_parameters(parameters)
        train_client(self.model, self.dataloader, epochs=5, lr=0.01)
        return self.get_parameters(), len(self.dataloader), {}

    def evaluate(self, parameters, config):
        """Evaluate the global model using the car's local data."""
        self.set_parameters(parameters)
        accuracy = test_model(self.model, self.dataloader)
        return float(accuracy), len(self.dataloader), {"accuracy": accuracy}



from flwr.common import Context
def client_fn(context):
    partition_id = int(context.node_config["partition-id"])
    car = grid.cars[partition_id][0]  # Access the car by its ID
    dataloader = client_loaders_with_overlap[partition_id]
    return CarClient(car, dataloader)
