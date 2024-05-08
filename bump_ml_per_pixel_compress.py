import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import bump_data_process as bdp

INNER_LAYER_SIZE = 64

def custom_init(tensor):
    # Your custom initialization logic here
    # For example, initialize with random values from a normal distribution
    return init.normal_(tensor, mean=0, std=0.1)

class BumpmapInfoNet(nn.Module):
    def __init__(self):
        super(BumpmapInfoNet, self).__init__()
        self.bias1 = nn.Parameter(custom_init(torch.empty(INNER_LAYER_SIZE)))
        self.fc1 = nn.Linear(2, INNER_LAYER_SIZE)  # Input layer (theta, phi) -> 64 nodes
        self.fc2 = nn.Linear(INNER_LAYER_SIZE, 1)  # Output layer (dist)

    def forward(self, x):
        x = self.fc1(x) + self.bias1
        x = torch.sin(x)
        x = torch.flatten(torch.relu(self.fc2(x)))
        return x

def createTensorBuffer(buffer_size, i, j, sample_data, device):
    sample_phi = np.empty(buffer_size, dtype = np.float32)
    sample_theta = np.empty(buffer_size, dtype = np.float32)
    sample_dist = np.empty(buffer_size, dtype = np.float32)

    # preparing data for ml.
    #print("start creating ml input and target data")
    index = 0
    for k in range(bdp.SAMPLE_RAY_COUNT):
        for s in range(bdp.ANGLE_SAMPLE_COUNT):
            sample_phi[index] = k
            sample_theta[index] = s
            sample_dist[index] = sample_data[i][j][k][s]
            index = index + 1

    inputs = torch.tensor(np.column_stack((sample_phi, sample_theta)), dtype=torch.float32).to(device)
    targets = torch.tensor(sample_dist, dtype=torch.float32).to(device)
    #print("end of data preparition")
    
    return inputs, targets

def trainMLModel(inputs, targets, buffer_size, file_name, write_out, device):
    batch_size  = 16 * 1024
    if buffer_size < 16*1024:
        batch_size = buffer_size

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BumpmapInfoNet().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 1000
    loss = None
    num_batches = buffer_size // batch_size
    early_stop = False
    for epoch in range(epochs):
        #print(f'Epoch {epoch+1}/{epochs}')
        if early_stop:
            print("epoch", epoch)
            break

        for batch_index, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss is not None and loss.item() < 0.0001:
                early_stop = True
                break

        #print(f'Epoch [{epoch+1}/{epochs}], Batch : [{batch_index}/{num_batches}], Loss: {loss.item():.5f}')
    
    if not early_stop:
        print("loss:", loss.item())

    if write_out:
        # Define a file path where you want to save your model
        output_path = file_name

        # Save the model
        torch.save(model.state_dict(), output_path)
    
    return model

def testMLModel(model, phi, theta, device):
    # Generate test spherical coordinates
    test_sample_phi = np.empty(1, dtype = np.float32)
    test_sample_theta = np.empty(1, dtype = np.float32)

    # preparing data for ml.
    #print("start creating ml test input data")
    k = int(phi / 360.0 * (bdp.SAMPLE_RAY_COUNT - 1))
    s = int(theta / 90.0 * (bdp.ANGLE_SAMPLE_COUNT - 1))
    test_sample_phi[0] = k
    test_sample_theta[0] = s
    #print("end creating ml test input data")

    # Convert to PyTorch tensor
    inputs_test = torch.tensor(np.column_stack((test_sample_phi, test_sample_theta)), dtype=torch.float32).to(device)

    with torch.no_grad():  # Inference mode, no need to calculate gradients
        predictions = model(inputs_test).cpu()

    return predictions.numpy().flatten()