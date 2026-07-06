import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.priors import PriorSampler
from src.waveform import generate_waveform
from src.noise import NoiseGenerator
from src.whitening import Whitening


class DatasetGenerator:

    def __init__(self, n_samples=1000, output_dir="data"):

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

        for _ in tqdm(
            range(self.n_samples),
            desc="Generating Dataset",
            ncols=100
        ):

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

            except Exception:
                continue

        waveforms = np.array(waveforms, dtype=np.float32)
        parameters = np.array(parameters, dtype=np.float32)

        np.save(self.output_dir / "waveforms.npy", waveforms)
        np.save(self.output_dir / "parameters.npy", parameters)

        print("\nDataset generation complete!")

        print("Waveforms:", waveforms.shape)
        print("Parameters:", parameters.shape)

        return waveforms, parameters