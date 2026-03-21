"""Minimal example: predict binaural audio quality with BIHAQ."""

from bihaq import load_model, predict

# Load model with pre-trained weights
model = load_model("weights/bihaq.pt")

# Score a degraded file against its reference
score = predict(model, "reference.wav", "degraded.wav")
print(f"Quality score: {score:.4f}")
