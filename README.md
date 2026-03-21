# Binaural Hearing Aid Music Quality

A learned, full-reference metric for **binaural (stereo) audio quality**.
Given a stereo reference and a degraded version, BIHAQ returns a score in
`[0, 1]` that correlates with human perceptual ratings.

## Architecture

BIHAQ uses two parallel branches:

| Branch | Input | Feature extraction | Temporal model |
|---|---|---|---|
| **Monaural** | STFT magnitude spectrogram (mid-channel delta) | 2-D CNN | Bi-LSTM |
| **Binaural** | ILD, IPD, IACC from gammatone filterbank (delta) | -- | Bi-LSTM |

Both branches are pooled via gated attention and concatenated into a dense
network with sigmoid output.

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10, PyTorch >= 2.2, and torchaudio >= 2.2.

## Quick start

```python
from bihaq import load_model, predict

model = load_model("weights/bihaq.pt")

score = predict(model, "reference.wav", "degraded.wav")
print(f"Quality: {score:.4f}")
```

With tensors (resampled automatically when `input_sample_rate` is given):

```python
import torchaudio
from bihaq import load_model, predict

model = load_model("weights/bihaq.pt")

ref, sr = torchaudio.load("reference.wav")
deg, _  = torchaudio.load("degraded.wav")

score = predict(model, ref, deg, input_sample_rate=sr)
```

Or calling the model directly for full control:

```python
import torch
import torchaudio
from bihaq import load_model, SAMPLE_RATE

model = load_model("weights/bihaq.pt")

ref, sr = torchaudio.load("reference.wav")
deg, _  = torchaudio.load("degraded.wav")

# Resample to 32 kHz if needed
if sr != SAMPLE_RATE:
    resample = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    ref, deg = resample(ref), resample(deg)

# Add batch dimension: (2, samples) -> (1, 2, samples)
ref, deg = ref.unsqueeze(0), deg.unsqueeze(0)

with torch.no_grad():
    out = model(ref, deg)

print(out["fused"])   # main quality score  (batch,)
print(out["mon"])     # monaural branch     (batch,)
print(out["bin"])     # binaural branch     (batch,)
```

## Input requirements

- **Stereo** audio (2 channels).
- **Hearing loss:** the degraded signal should have hearing loss simulated. The model is not designed to apply hearing loss simulation itself (yet)
- **Sample rate:** 32 kHz (the `predict` helper resamples automatically; if
  calling `model.forward` directly, resample beforehand).
- Both reference and degraded signals should be **time-aligned** and the
  **same length**.

## Project structure

```
bihaq/
  __init__.py      # load_model(), predict()
  model.py         # BIHAQ, GatedAttentionPooling
  encoders.py      # MonEncoder, SpectrogramDeltaCNN, BinEncoder
  binaural.py      # BinauralMetrics (ILD, IPD, IACC)
  dsp.py           # GammatoneFilterBank
examples/
  predict.py       # Minimal usage example
  export_weights.py
weights/
  bihaq.pt
```


## License

MIT
