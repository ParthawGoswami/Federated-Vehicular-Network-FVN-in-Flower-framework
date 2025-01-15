from flwr.client import NumPyClient
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from collections import OrderedDict
import struct
import json
import math
import unittest
import pandas as pd
import os
import jax
from jax import grad
from torch.utils.data import DataLoader
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the SimpleNet model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten CIFAR-10 images
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function for each car
def train_car(car, data, targets, global_model_params, lr=0.1):
    car.model.load_state_dict(global_model_params)
      
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(car.model.parameters(), lr=lr)

    car.model.train()
    optimizer.zero_grad()
        
    outputs = car.model(data)
    loss = criterion(outputs, targets)
    loss.backward()

    gradients = {name: param.grad.clone() for name, param in car.model.named_parameters() if param.grad is not None}
    optimizer.step()
    return gradients

# Evaluate model accuracy
def evaluate_model(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        total = test_labels.size(0)
        correct = (predicted == test_labels).sum().item()
    return 100 * correct / total

# Define the Grid class
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
        self.empty_spaces = set((i, j) for i in range(rows) for j in range(cols))
        self.cars = []

    def add_car(self, car, position):
        if (position[0] < 0 or position[0] >= self.rows or position[1] < 0 or position[1] >= self.cols):
            raise IndexError("Position out of bounds")
        if position in self.empty_spaces:
            self.grid[position[0]][position[1]] = car
            self.empty_spaces.remove(position)
            self.cars.append((car, position))
            car.position = position
        else:
            print("Position already occupied or out of bounds")

    def move_car(self, car, new_position):
        if (new_position[0] < 0 or new_position[0] >= self.rows or new_position[1] < 0 or new_position[1] >= self.cols):
            raise IndexError("New position out of bounds")
        if new_position in self.empty_spaces:
            old_position = car.position
            self.grid[old_position[0]][old_position[1]] = None
            self.grid[new_position[0]][new_position[1]] = car
            self.empty_spaces.add(old_position)
            self.empty_spaces.remove(new_position)
            car.position = new_position
        else:
            print("New position already occupied or out of bounds")

    def remove_car(self, car):
        if car in [car for car, _ in self.cars]:
            position = car.position
            self.grid[position[0]][position[1]] = None
            self.empty_spaces.add(position)
            self.cars.remove((car, position))
        else:
            raise ValueError("Car not found in the grid")

    def add_random_cars(self, num_cars):
        for _ in range(num_cars):
            car_id = len(self.cars)
            car = Car(car_id)
            position = random.choice(list(self.empty_spaces))
            self.add_car(car, position)

# Define the Antenna class
class Antenna:
    DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    def __init__(self):
        self.direction_index = 2  # Default to 'E'

    def rotate_clockwise(self):
        self.direction_index = (self.direction_index + 1) % 8

    def rotate_counter_clockwise(self):
        self.direction_index = (self.direction_index - 1) % 8

    def get_direction(self):
        return Antenna.DIRECTIONS[self.direction_index]

# Define the number of cars (clients) and the number of samples per client
num_clients = 10
samples_per_client = 5000

# Ensure the CIFAR-10 training dataset has enough samples
assert len(train_images_tensor) == num_clients * samples_per_client, \
    "The dataset size does not match the expected split."

# Split the dataset into 10 distinct parts
car_data_splits = torch.chunk(train_images_tensor, num_clients)
car_label_splits = torch.chunk(train_labels_tensor, num_clients)
    
    
# Define the Car class
class Car:
    def __init__(self, id, data, targets, antenna_range=3):
        self.id = id
        self.antenna = Antenna()
        self.position = None
        self.data_history = []
        self.antenna_range = antenna_range
        self.model = SimpleNet()  # Each car has its own model
        #self.data = train_images_tensor  # Using MNIST train images
        #self.targets = train_labels_tensor  # Using MNIST train labels
        self.data = data  # Assign distinct data to each car
        self.targets = targets  # Assign distinct labels to each car

    def receive_data(self, data):
        self.data_history.append(data)

    def __repr__(self):
        return f"Car(id={self.id}, range={self.antenna_range})"

# Define the Simulation class
class Simulation:
    def __init__(self, grid, path_loss_probability=0.2):
        self.grid = grid
        self.base_path_loss_probability = path_loss_probability
        self.active_cars = set()
        self.inactive_cars = set()
        self.global_model = SimpleNet()
        self.server_car = None

    def set_server_car(self, car):
        self.server_car = car

    def move_car_randomly(self):
        car, old_position = random.choice(self.grid.cars)
        new_position = random.choice(list(self.grid.empty_spaces))
        self.grid.move_car(car, new_position)

    def send_data(self, sender, payload, path_loss_probability=None):
        if path_loss_probability is None:
            path_loss_probability = self.base_path_loss_probability
        direction = sender.antenna.get_direction()
        receivers = self.get_receivers(sender, direction)
        receiving_cars = []
        for receiver in receivers:
            distance = self.calculate_distance(sender.position, receiver.position)
            effective_path_loss_probability = self.adjust_path_loss_probability(distance, path_loss_probability)
            if random.random() > effective_path_loss_probability:
                receiver.receive_data(payload)
                receiving_cars.append(receiver)
                print(f"Data sent from Car {sender.id} to Car {receiver.id}")
            else:
                print(f"Data loss from Car {sender.id} to Car {receiver.id}")
        return receiving_cars

    def aggregate_gradients(self, gradients_list):
        aggregated_gradients = {}
        for gradients in gradients_list:
            for name, grad in gradients.items():
                if name not in aggregated_gradients:
                    aggregated_gradients[name] = grad.clone()
                else:
                    aggregated_gradients[name] += grad
        return aggregated_gradients

    def update_global_model(self, aggregated_gradients, lr=0.1):
        if not hasattr(self, 'optimizer'):
            self.optimizer = optim.SGD(self.global_model.parameters(), lr=lr)

        # Manually set gradients
        for name, param in self.global_model.named_parameters():
            if name in aggregated_gradients:
                if param.grad is None:
                    param.grad = aggregated_gradients[name] / len(aggregated_gradients)
                else:
                    param.grad.data.copy_(aggregated_gradients[name] / len(aggregated_gradients))

        self.optimizer.step()
        self.optimizer.zero_grad()  # Clear the gradients after the optimizer step

    def train_step(self, lr=0.01):
        if not self.server_car:
            raise ValueError("Server car not set")

        gradients_list = []
        for car, _ in self.grid.cars:
            gradients = train_car(car, car.data, car.targets, self.global_model.state_dict(), lr=lr)
            gradients_list.append(gradients)

        aggregated_gradients = self.aggregate_gradients(gradients_list)
        self.update_global_model(aggregated_gradients, lr=lr)

        for car, _ in self.grid.cars:
            car.model.load_state_dict(self.global_model.state_dict())

    def get_receivers(self, sender, direction):
        receivers = []
        row, col = sender.position
        max_range = sender.antenna_range
        if direction == 'N':
            for i in range(1, max_range + 1):
                if row - i >= 0 and self.grid.grid[row - i][col]:
                    receivers.append(self.grid.grid[row - i][col])
        elif direction == 'NE':
            for i in range(1, max_range + 1):
                if row - i >= 0 and col + i < self.grid.cols and self.grid.grid[row - i][col + i]:
                    receivers.append(self.grid.grid[row - i][col + i])
        elif direction == 'E':
            for j in range(1, max_range + 1):
                if col + j < self.grid.cols and self.grid.grid[row][col + j]:
                    receivers.append(self.grid.grid[row][col + j])
        elif direction == 'SE':
            for i in range(1, max_range + 1):
                if row + i < self.grid.rows and col + i < self.grid.cols and self.grid.grid[row + i][col + i]:
                    receivers.append(self.grid.grid[row + i][col + i])
        elif direction == 'S':
            for i in range(1, max_range + 1):
                if row + i < self.grid.rows and self.grid.grid[row + i][col]:
                    receivers.append(self.grid.grid[row + i][col])
        elif direction == 'SW':
            for i in range(1, max_range + 1):
                if row + i < self.grid.rows and col - i >= 0 and self.grid.grid[row + i][col - i]:
                    receivers.append(self.grid.grid[row + i][col - i])
        elif direction == 'W':
            for j in range(1, max_range + 1):
                if col - j >= 0 and self.grid.grid[row][col - j]:
                    receivers.append(self.grid.grid[row][col - j])
        elif direction == 'NW':
            for i in range(1, max_range + 1):
                if row - i >= 0 and col - i >= 0 and self.grid.grid[row - i][col - i]:
                    receivers.append(self.grid.grid[row - i][col - i])
        print(f"Receivers for direction {direction} from position ({row}, {col}): {[car.id for car in receivers]}")
        return receivers

    def calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def adjust_path_loss_probability(self, distance, base_path_loss_probability):
        return base_path_loss_probability * (1 + 0.1 * distance)

    def run_step(self, num_transmitters=1):
        self.move_car_randomly()
        self.active_cars.clear()
        self.inactive_cars.clear()

        # Select num_transmitters cars to transmit
        transmitting_cars = random.sample([car for car, _ in self.grid.cars], min(num_transmitters, len(self.grid.cars)))

        for car, _ in self.grid.cars:
            if car in transmitting_cars:
                car.antenna.rotate_clockwise()
                receiving_cars = self.send_data(car, {"payload": "data"})
                self.active_cars.add(car)
                print(f"Active Car: {car.id}, Receiving Cars: {[c.id for c in receiving_cars]}")
            else:
                self.inactive_cars.add(car)

    def save_configuration(self, filename):
        config = {
            "cars": [{"id": car.id, "position": car.position, "data_history": car.data_history, "antenna_range": car.antenna_range} for car, _ in self.grid.cars],
            "empty_spaces": list(self.grid.empty_spaces)
        }
        with open(filename, 'w') as f:
            json.dump(config, f)

    def load_configuration(self, filename):
        with open(filename, 'r') as f:
            config = json.load(f)
        self.grid.grid = [[None for _ in range(self.grid.cols)] for _ in range(self.grid.rows)]
        self.grid.empty_spaces = set((i, j) for i in range(self.grid.rows) for j in range(self.grid.cols))
        self.grid.cars = []
        for car_data in config['cars']:
            car = Car(car_data['id'], car_data['antenna_range'])
            car.position = tuple(car_data['position'])
            car.data_history = car_data['data_history']
            self.grid.grid[car.position[0]][car.position[1]] = car
            self.grid.empty_spaces.remove(car.position)
            self.grid.cars.append((car, car.position))

# Run simulation with MNIST data
import csv

# Initialize the grid and simulation, and add cars with unique datasets
grid = Grid(5, 5)
simulation = Simulation(grid)

# Add 10 cars to the grid, each with a unique subset of data
for i in range(num_clients):
    car = Car(i, car_data_splits[i], car_label_splits[i])
    position = random.choice(list(grid.empty_spaces))
    grid.add_car(car, position)

simulation.set_server_car(simulation.grid.cars[0][0])  # Set the first car as the server



