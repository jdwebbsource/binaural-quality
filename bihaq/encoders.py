"""Monaural and binaural encoders for the BIHAQ model."""

import torch
import torch.nn as nn

from bihaq.binaural import BinauralMetrics


class MonEncoder(nn.Module):
    """Monaural encoder using STFT magnitude spectrograms.

    Computes separate magnitude spectrograms for the left and right
    channels, returned as a tuple for downstream mid-channel delta
    processing.

    Args:
        n_fft: FFT size.
        hop_size: STFT hop length in samples (1024 at 44.1 kHz).
        input_batchnorm: Apply batch normalisation to spectrograms.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_size: int = 1024,
        input_batchnorm: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.register_buffer("window", torch.hann_window(n_fft))
        self.bn = nn.BatchNorm2d(1) if input_batchnorm else nn.Identity()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Stereo audio ``(batch, 2, samples)``.

        Returns:
            Tuple ``(left_mag, right_mag)``, each ``(batch, freq_bins, time)``.
        """
        assert x.shape[1] == 2, "Input must be stereo audio."

        X_l = torch.abs(
            torch.stft(
                x[:, 0],
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                window=self.window,
                return_complex=True,
            )
        )
        X_r = torch.abs(
            torch.stft(
                x[:, 1],
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                window=self.window,
                return_complex=True,
            )
        )
        X_l = self.bn(X_l.unsqueeze(1)).squeeze(1)
        X_r = self.bn(X_r.unsqueeze(1)).squeeze(1)
        return X_l, X_r


class SpectrogramDeltaCNN(nn.Module):
    """2-D CNN that extracts frame-wise features from a magnitude spectrogram.

    Input shape:  ``(batch, freq_bins, time)``
    Output shape: ``(batch, time, out_dim)``

    Args:
        in_channels: Number of input channels (1 for magnitude).
        base_channels: Base number of convolutional filters.
        out_dim: Output feature dimension per frame.
        use_batchnorm: Use batch normalisation in conv blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        out_dim: int = 64,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim

        def _norm(c):
            return nn.BatchNorm2d(c) if use_batchnorm else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            _norm(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            _norm(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            _norm(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Conv2d(base_channels * 4, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (bs, 1, freq, time)
        x = self.conv(x)
        x = self.pool(x)            # (bs, C, 1, time)
        x = self.proj(x)            # (bs, out_dim, 1, time)
        x = x.squeeze(2)            # (bs, out_dim, time)
        return x.permute(0, 2, 1)   # (bs, time, out_dim)


class BinEncoder(nn.Module):
    """Binaural encoder using gammatone filterbank.

    Computes ILD, IPD, and IACC from stereo audio and returns raw binaural
    features ``(3 * num_filts)`` per frame without a learned projection.

    Args:
        num_filts: Number of gammatone filters.
        frame_len: Analysis frame length in samples.
        hop_size: Hop size between frames in samples.
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(
        self,
        num_filts: int = 20,
        frame_len: int = 2048,
        hop_size: int = 1024,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.num_filts = num_filts
        self.BinauralModel = BinauralMetrics(
            frame_len=frame_len,
            hop_size=hop_size,
            sample_rate=sample_rate,
            filter_num=num_filts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stereo audio ``(batch, 2, samples)``.

        Returns:
            Features ``(batch, num_frames, 3 * num_filts)``.
        """
        features = self.BinauralModel(x)       # (bs, 3*num_filts, num_frames)
        return features.permute(0, 2, 1)       # (bs, num_frames, 3*num_filts)
