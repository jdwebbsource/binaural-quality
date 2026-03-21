"""BIHAQ -- Binaural Audio Quality prediction."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from bihaq.model import BIHAQ, GatedAttentionPooling
from bihaq.encoders import (
    MonEncoder,
    SpectrogramDeltaCNN,
    BinEncoder,
)
from bihaq.binaural import BinauralMetrics
from bihaq.dsp import GammatoneFilterBank

__all__ = [
    "load_model",
    "predict",
    "BIHAQ",
    "GatedAttentionPooling",
    "MonEncoder",
    "SpectrogramDeltaCNN",
    "BinEncoder",
    "BinauralMetrics",
    "GammatoneFilterBank",
    "SAMPLE_RATE",
]

SAMPLE_RATE = 32_000


def load_model(
    weights: str | Path | None = None,
    map_location: str = "cpu",
) -> BIHAQ:
    """Build the BIHAQ model and optionally load pre-trained weights.

    Args:
        weights: Path to a ``.pt`` state-dict file **or** a PyTorch-Lightning
            ``.ckpt`` checkpoint.  When ``None`` the model is returned with
            random initialisation.
        map_location: Device string passed to :func:`torch.load`.

    Returns:
        A :class:`BIHAQ` model in eval mode.
    """
    model = BIHAQ(
        mon_encoder=MonEncoder(n_fft=2048, hop_size=2048),
        mon_cnn=SpectrogramDeltaCNN(base_channels=32, out_dim=64),
        bin_encoder=BinEncoder(
            num_filts=20,
            frame_len=1024,
            hop_size=1024,
            sample_rate=SAMPLE_RATE,
        ),
        mon_cnn_out_dim=64,
        num_filts=20,
        mon_lstm_size=32,
        bin_lstm_size=64,
        lstm_layers=1,
        dense_dropout=0.15,
    )

    if weights is not None:
        weights = Path(weights)
        # Lightning .ckpt files may contain non-tensor objects
        weights_only = weights.suffix != ".ckpt"
        state = torch.load(
            weights, map_location=map_location, weights_only=weights_only,
        )
        if "state_dict" in state:
            state = {
                k.replace("model.", "", 1): v
                for k, v in state["state_dict"].items()
                if k.startswith("model.")
            }
        # strict=False: gammatone filter coefficients are deterministic
        # buffers that are recomputed from config, so they won't be in
        # checkpoints from older code that didn't register them.
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


def predict(
    model: BIHAQ,
    ref: str | Path | torch.Tensor,
    deg: str | Path | torch.Tensor,
    input_sample_rate: int | None = None,
) -> float:
    """Score a single reference / degraded pair.

    Args:
        model: A :class:`BIHAQ` instance (from :func:`load_model`).
        ref: File path **or** tensor ``(2, samples)`` of stereo reference audio.
        deg: File path **or** tensor ``(2, samples)`` of stereo degraded audio.
        input_sample_rate: Sample rate of the input when passing tensors.
            If the rate differs from :data:`SAMPLE_RATE` (32 kHz) the audio
            is resampled automatically.  Ignored when loading from file paths
            (the file's native rate is read instead).

    Returns:
        Quality score in ``[0, 1]`` (higher is better).
    """

    def _prepare(src):
        if isinstance(src, (str, Path)):
            wav, sr = torchaudio.load(str(src))
            if sr != SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
            return wav
        if input_sample_rate is not None and input_sample_rate != SAMPLE_RATE:
            src = torchaudio.transforms.Resample(
                input_sample_rate, SAMPLE_RATE,
            )(src)
        return src

    ref_t = _prepare(ref).unsqueeze(0)
    deg_t = _prepare(deg).unsqueeze(0)

    device = next(model.parameters()).device
    ref_t, deg_t = ref_t.to(device), deg_t.to(device)

    with torch.no_grad():
        return model(ref_t, deg_t)["fused"].item()
