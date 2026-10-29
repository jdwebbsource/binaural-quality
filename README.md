# Binaural Hearing Aid Music Quality

A learned, full-reference metric for **binaural audio quality**.
Given a stereo reference and a degraded version, BIHAQ returns a score in
`[0, 1]` that correlates with human perceptual ratings of music processed
through simulated hearing aids.

Reported paper metrics use five-fold **out-of-fold** models. The checkpoint
in `weights/bihaq.pt` is a separate network trained on **all labelled items**.

## Architecture

BIHAQ uses two parallel branches (44.1 kHz):

| Branch | Input | Feature extraction | Temporal model |
|---|---|---|---|
| **Monaural** | STFT magnitude spectrogram (mid-channel delta, n_fft=2048, hop=1024) | 2-D CNN | Bi-LSTM |
| **Binaural** | ILD, IPD, IACC from a 20-band gammatone filterbank (frame=2048, hop=1024) | -- | Bi-LSTM |

Both branches are pooled via gated attention and concatenated into a dense
network with sigmoid output.

## Installation

```bash
pip install -e .
```
or via uv
```bash
uv sync
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

if sr != SAMPLE_RATE:
    resample = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
    ref, deg = resample(ref), resample(deg)

ref, deg = ref.unsqueeze(0), deg.unsqueeze(0)

with torch.no_grad():
    out = model(ref, deg)

print(out["fused"])   # main quality score  (batch,)
print(out["mon"])     # monaural branch     (batch,)
print(out["bin"])     # binaural branch     (batch,)
```

## Input requirements

- **Stereo** audio (2 channels).
- **Hearing loss:** the degraded signal should have hearing loss simulated. The model is not designed to apply hearing loss simulation itself.
- **Sample rate:** 44.1 kHz (the `predict` helper resamples automatically; if
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
  predict.py
weights/
  bihaq.pt         # model weights
```


## License

MIT
