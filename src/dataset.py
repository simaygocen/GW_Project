import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.priors import PriorSampler
from src.waveform import generate_waveform
from src.noise import NoiseGenerator
from src.whitening import Whitening


class DatasetGenerator:

    def __init__(self, n_samples=25000, output_dir="data"):

        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prior = PriorSampler()
        self.noise = NoiseGenerator()
        self.whitener = Whitening()

    def generate(self):

        waveforms = []
        parameters = []

        print(f"\nGenerating {self.n_samples} samples...\n")

        attempts = 0
        max_attempts = 2 * self.n_samples
        progress = tqdm(total=self.n_samples, desc="Generating Dataset", ncols=100)

        while len(waveforms) < self.n_samples and attempts < max_attempts:
            attempts += 1

            try:

                theta = self.prior.sample()

                waveform = generate_waveform(theta)

                noisy_waveform = self.noise.add_noise(waveform)

                whitened_waveform = self.whitener.whiten(noisy_waveform)

                waveforms.append(
                    whitened_waveform.astype(np.float32)
                )

                parameters.append(
                    theta.to_array()
                )

                progress.update(1)

            except Exception:
                continue

        progress.close()

        if len(waveforms) != self.n_samples:
            raise RuntimeError(
                f"Generated only {len(waveforms)} valid samples after {attempts} attempts"
            )

        waveforms = np.array(waveforms, dtype=np.float32)
        parameters = np.array(parameters, dtype=np.float32)

        dataset_path = self.output_dir / f"gw_dataset_{self.n_samples}.npz"
        np.savez(dataset_path, X=waveforms, theta=parameters)

        print("\nDataset generation complete!")

        print("Waveforms:", waveforms.shape)
        print("Parameters:", parameters.shape)
        print("Saved:", dataset_path)

        return waveforms, parameters
