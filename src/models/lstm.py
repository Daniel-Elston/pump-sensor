from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from utils.file_log import Logger


class LSTMAD(nn.Module):
    def __init__(self, config):
        self.config = config
        self.logger = Logger('MakeDatasetLog', f'{
                             Path(__file__).stem}.log').get_logger()
        super(LSTMAD, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,  # Single feature per time step
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=config['hidden_size'],
            out_features=1  # Output one value per time step
        )

    def forward(self, x):
        # print(f"Shape of input to LSTM: {x.shape}")
        lstm_out, _ = self.lstm(x)
        # print(f'Shape of lstm_out:{lstm_out.shape}')
        output = self.linear(lstm_out)
        return output

    def train_model(self, train_dataloader):
        """
        Train the model.
        Args:
            train_dataloader (DataLoader): Dataloader for the training set.
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config['learning_rate'])
        print('Training model...')
        for epoch in range(self.config['n_epochs']):
            total_loss = 0
            for seqs in train_dataloader:
                optimizer.zero_grad()
                output = self(seqs)
                # Target is the input sequence itself
                loss = criterion(output, seqs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            print(
                f'Epoch {epoch+1}/{self.config['n_epochs']} Loss: {avg_loss:.8f}')
        print('Finished training.')

    def detect_anomalies(self, test_dataloader):
        threshold = self.config['threshold']
        self.eval()
        anomalies = []
        with torch.no_grad():
            for seqs in test_dataloader:
                output = self(seqs)
                loss = torch.mean(torch.abs(output - seqs), dim=1)
                batch_anomalies = (loss > threshold).int().view(-1).tolist()
                anomalies.extend(batch_anomalies)
        return anomalies
