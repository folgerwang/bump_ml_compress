import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import bump_data_process as bdp

def custom_init(tensor):
    # Your custom initialization logic here
    # For example, initialize with random values from a normal distribution
    return init.normal_(tensor, mean=0, std=0.1)

class BumpmapInfoNet(nn.Module):
    def __init__(self):
        super(BumpmapInfoNet, self).__init__()
        self.bias1 = nn.Parameter(custom_init(torch.empty(256)))
        self.bias2 = nn.Parameter(custom_init(torch.empty(64)))
        self.fc1 = nn.Linear(2, 256)  # Input layer (theta, phi) -> 64 nodes
        self.fc2 = nn.Linear(256, 64)  # Input layer (theta, phi) -> 64 nodes
        self.fc3 = nn.Linear(64, 1)  # Output layer (dist)

    def forward(self, x):
        x = torch.sin(self.fc1(x) + self.bias1)
        x = torch.sin(self.fc2(x) + self.bias2)
        x = torch.flatten(torch.relu(self.fc3(x)))
        return x

def createTensorBuffer(buffer_size, w, h, sample_data, device):
    sample_xy = np.empty(buffer_size, dtype = np.float32)
    sample_phi_theta = np.empty(buffer_size, dtype = np.float32)
    sample_dist = np.empty(buffer_size, dtype = np.float32)

    # preparing data for ml.
    print("start creating ml input and target data")
    index = 0
    for i in range(w):
        print("line idx", i, " of ", w)
        for j in range(h):
            for k in range(bdp.SAMPLE_RAY_COUNT):
                for s in range(bdp.ANGLE_SAMPLE_COUNT):
                    sample_xy[index] = ((i<<2) * 1024 + (j<<2)) / float( 1024 * 1024)
                    sample_phi_theta[index] = ((k<<2) * 1024 + (s<<6)) / float(1024 * 1024)
                    sample_dist[index] = sample_data[i][j][k][s]
                    index = index + 1

    inputs = torch.tensor(np.column_stack((sample_xy, sample_phi_theta)), dtype=torch.float32).to(device)
    targets = torch.tensor(sample_dist, dtype=torch.float32).to(device)
    print("end of data preparition")
    
    return inputs, targets

def trainMLModel(inputs, targets, buffer_size, file_name, write_out, device):
    batch_size = 256*1024
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BumpmapInfoNet().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 1
    num_batches = buffer_size / batch_size
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for batch_index, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{epochs}], Batch : [{batch_index}/{num_batches}], Loss: {loss.item():.4f}')

    if write_out:
        # Define a file path where you want to save your model
        output_path = file_name

        # Save the model
        torch.save(model.state_dict(), output_path)

def testMLModel(model, w, h, phi, theta, device):
    # Generate test spherical coordinates
    num_test_samples = w * h  # Number of test samples

    test_sample_xy = np.empty(num_test_samples, dtype = np.float32)
    test_sample_phi_theta = np.empty(num_test_samples, dtype = np.float32)

    # preparing data for ml.
    print("start creating ml test input data")
    index = 0
    for i in range(w):
        print("line idx", i, " of ", w)
        for j in range(h):
            test_sample_xy[index] = ((i<<2) * 1024 + (j<<2)) / float(1024 * 1024)
            k = int(phi / 360.0 * (bdp.SAMPLE_RAY_COUNT - 1))
            s = int(theta / 90.0 * (bdp.ANGLE_SAMPLE_COUNT - 1))
            test_sample_phi_theta[index] = ((k<<2) * 1024 + (s<<6)) / float(1024 * 1024)
            index = index + 1
    print("end creating ml test input data")

    # Convert to PyTorch tensor
    inputs_test = torch.tensor(np.column_stack((test_sample_xy, test_sample_phi_theta)), dtype=torch.float32).to(device)

    with torch.no_grad():  # Inference mode, no need to calculate gradients
        predictions = model(inputs_test).cpu()

    return predictions.numpy().flatten()