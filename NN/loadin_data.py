from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch


def load_data(data : pd.DataFrame, target_column : list, batch_size : int):
    features = data.drop(columns=target_column).values
    target = data[target_column].values
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Note shuffle=False for validation

    return train_loader, val_loader,X_train, X_test, y_train, y_test

class CustomDataset():
    def __init__(self, dataframe, target_column, transform=None):

        self.dataframe = dataframe
        self.target_column = target_column
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract features and target from the DataFrame
        features = self.dataframe.drop(columns=self.target_column).values  # Exclude the target column
        target = self.dataframe.loc[idx, self.target_column]  # Use the specified target column
        
        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.long)  # Adjust the dtype as needed
        
        if self.transform:
            features = self.transform(features)

        return features, target

def create_final_val_loader(dataframe, target_column, batch_size=32, num_workers=0, transform=None):
    val_dataset = CustomDataset(dataframe, target_column, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    return val_loader
