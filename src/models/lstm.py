from __future__ import annotations

import torch
import torch.nn as nn


class LSTMAD(nn.Module):
    def __init__(self, config):
        super(LSTMAD, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config['n_features'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True
        )
        # Adjusted to output the hidden_size
        self.linear = nn.Linear(
            in_features=config['hidden_size'],
            out_features=config['n_features']
        )

    def forward(self, x):
        """
        Forward pass through the model.
        x: (batch_size, seq_length, n_features)
        """
        lstm_out, _ = self.lstm(x)
        # The last output of the sequence is considered
        return self.linear(lstm_out[:, -1, :])

    def train_model(self, train_dataloader):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config['learning_rate'])

        for epoch in range(self.config['epochs']):
            total_loss = 0
            for seqs, _ in train_dataloader:
                optimizer.zero_grad()
                output = self(seqs)
                # Comparing with the last item in sequence
                loss = criterion(output, seqs[:, -1, :])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            print(
                f'Epoch {epoch+1}/{self.config["epochs"]} Loss: {avg_loss:.8f}')

        # Save the model after training
        torch.save(self.state_dict(), self.config['model_path'])

    def detect_anomalies(self, test_dataloader, threshold=0.1):
        self.eval()
        anomalies = []
        with torch.no_grad():
            for seqs, _ in test_dataloader:
                output = self(seqs)
                loss = torch.mean(torch.abs(output - seqs[:, -1, :]), dim=1)
                anomalies.extend(loss > threshold)
        return anomalies
