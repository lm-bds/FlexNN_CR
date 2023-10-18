import pandas as pd
import numpy as np




import torch
import torch.nn as nn




class DLearnModel(nn.Module):
    def __init__(self, input_size, hidden_units=None, activation_function='ReLU', dropout_rate=0.5):
        super(DLearnModel, self).__init__()
        
        if hidden_units is None:
            # Default hidden_units if not provided
            hidden_units = [64, 32]  # Modify the default architecture as needed

        # Define the layers and architecture of your model based on the hidden_units and activation_function
        self.layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_units[i]))
            else:
                self.layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            if activation_function == 'ReLU':
                self.layers.append(nn.ReLU())
            elif activation_function == 'Tanh':
                self.layers.append(nn.Tanh())
            elif activation_function == 'Sigmoid':
                self.layers.append(nn.Sigmoid())
            # Add dropout layer with the specified dropout_rate
            #self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(hidden_units[-1], 1))  # Output layer
        
        # Create a Sequential model to stack the layers
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)




def validation_step(model, criterion, batch_x, batch_y):
    # Set the model in evaluation mode
    model.eval()

    # Disable gradient computation during validation
    with torch.no_grad():
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

    return loss.item()

def validate(model, criterion, val_loader):
    # Initialize validation loss
    val_loss = 0.0

    # Loop through the validation dataset
    for batch_x, batch_y in val_loader:
        # Perform a single validation step
        loss = validation_step(model, criterion, batch_x, batch_y)
        val_loss += loss

    # Calculate and return the average validation loss
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def train_step(model, optimizer, criterion, batch_x, batch_y, clip_gradients=False, max_norm=None):
    # Set the model in training mode
    model.train()

    # Forward pass
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()

    if clip_gradients:
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    optimizer.step()

    # Return loss and gradients (optional)
    if clip_gradients:
        return loss.item(), torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    else:
        return loss.item(), None


    
