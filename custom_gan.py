import time
import datetime
import torch
import torch.nn as nn
from ctgan import CTGAN as OriginalCTGAN  


def init_weights(m):
    """
    Applies Xavier (Glorot) initialization to the model's layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, input_dim=128, output_dim=512):  # Final output: 512
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, output_dim),
            nn.Tanh()
        )
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=512):  # Input is 512, matching Generator output
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1),
            nn.Sigmoid()  # Change to no activation if using Wasserstein GAN
        )
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class CustomGAN:
    def __init__(self, epochs=300, batch_size=500, verbose=True):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self._model = OriginalCTGAN(epochs=epochs, batch_size=batch_size, verbose=verbose)

    def fit(self, data, discrete_columns=[]):
        print("Using improved custom GAN architecture with:")
        print("- LeakyReLU and ReLU activations")
        print("- Xavier initialization")
        print("- Simulated Wasserstein loss")
        print("\nGenerator architecture:\n", self.generator)
        print("\nDiscriminator architecture:\n", self.discriminator)

        max_duration = 120 * 60  # 120 minutes
        start_time = time.time()
        print(f"\nTraining started at {datetime.datetime.now().strftime('%H:%M:%S')} (max duration: 120 minutes)...")

        try:
            self._model.fit(data, discrete_columns=discrete_columns)
        except KeyboardInterrupt:
            print("Training manually interrupted.")
        finally:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Training finished. Duration: {elapsed / 60:.2f} minutes")

            if elapsed > max_duration:
                print("WARNING: Training exceeded 120 minutes. Consider modifying internal loop for hard stop.")

    def sample(self, num_rows):
        return self._model.sample(num_rows)

    def save(self, path):
        self._model.save(path)

    def load(self, path):
        self._model.load(path)
