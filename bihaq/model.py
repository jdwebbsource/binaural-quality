"""BIHAQ: binaural hearing aid quality prediction."""

import torch
import torch.nn as nn


class GatedAttentionPooling(nn.Module):
    """Gated attention pooling over a temporal sequence.

    Computes an attention-weighted, feature-gated summary of a
    variable-length sequence into a fixed-size vector.

    Args:
        input_dim: Feature dimension of input sequence.
        attn_hidden: Hidden dimension for the attention scorer.
    """

    def __init__(self, input_dim: int, attn_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: ``(batch, time, features)``.
            mask: Optional ``(batch, time)`` boolean mask for padded frames.

        Returns:
            Pooled vector ``(batch, features)``.
        """
        attn_scores = self.attn(x).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        gated_x = self.gate(x) * x
        return torch.bmm(attn_weights.unsqueeze(1), gated_x).squeeze(1)


class BIHAQ(nn.Module):
    """Binaural Hearing Aid Quality (BIHAQ).

    Predicts a perceptual quality score in ``[0, 1]`` for a degraded stereo
    audio signal relative to a reference, using parallel monaural and
    binaural processing branches.

    **Monaural branch:** STFT magnitude spectrograms -> mid-channel delta ->
    2-D CNN -> bidirectional LSTM -> gated attention pooling.

    **Binaural branch:** gammatone filterbank -> ILD / IPD / IACC deltas ->
    bidirectional LSTM -> gated attention pooling.

    The pooled representations are concatenated and fed through a dense
    network with sigmoid output.

    Args:
        mon_encoder: Monaural encoder (e.g. :class:`MonEncoder`).
        mon_cnn: CNN for spectrogram features (e.g. :class:`SpectrogramDeltaCNN`).
        bin_encoder: Binaural encoder (e.g. :class:`BinEncoder`).
        mon_cnn_out_dim: Feature dim produced by *mon_cnn*.
        num_filts: Number of gammatone filters in the binaural encoder.
        mon_lstm_size: Hidden size per direction for the monaural LSTM.
        bin_lstm_size: Hidden size per direction for the binaural LSTM.
        lstm_layers: Number of LSTM layers.
        dense_dropout: Dropout rate for the dense head.
    """

    def __init__(
        self,
        mon_encoder: nn.Module,
        mon_cnn: nn.Module,
        bin_encoder: nn.Module,
        mon_cnn_out_dim: int = 64,
        num_filts: int = 20,
        mon_lstm_size: int = 32,
        bin_lstm_size: int = 64,
        lstm_layers: int = 1,
        dense_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.mon_encoder = mon_encoder
        self.mon_cnn = mon_cnn
        self.bin_encoder = bin_encoder

        mon_input_dim = mon_cnn_out_dim
        bin_input_dim = 3 * num_filts

        self.mon_lstm = nn.LSTM(
            input_size=mon_input_dim,
            hidden_size=mon_lstm_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.bin_lstm = nn.LSTM(
            input_size=bin_input_dim,
            hidden_size=bin_lstm_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.mon_pool = GatedAttentionPooling(2 * mon_lstm_size)
        self.bin_pool = GatedAttentionPooling(2 * bin_lstm_size)

        self.mon_norm = nn.LayerNorm(2 * mon_lstm_size)
        self.bin_norm = nn.LayerNorm(2 * bin_lstm_size)

        pool_dim = 2 * mon_lstm_size + 2 * bin_lstm_size
        self.pre_dense_norm = nn.LayerNorm(pool_dim)

        self.dense = nn.Sequential(
            nn.Linear(pool_dim, int(pool_dim * 0.75)),
            nn.ReLU(),
            nn.Dropout(dense_dropout),
            nn.Linear(int(pool_dim * 0.75), pool_dim // 2),
            nn.ReLU(),
            nn.Dropout(dense_dropout),
            nn.Linear(pool_dim // 2, pool_dim // 4),
            nn.ReLU(),
            nn.Dropout(dense_dropout),
            nn.Linear(pool_dim // 4, 32),
            nn.ReLU(),
            nn.Dropout(dense_dropout * 0.5),
            nn.Linear(32, 1),
        )

        self.mon_head = nn.Sequential(
            nn.Linear(2 * mon_lstm_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.bin_head = nn.Sequential(
            nn.Linear(2 * bin_lstm_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        ref: torch.Tensor,
        deg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict quality scores.

        Args:
            ref: Reference stereo audio ``(batch, 2, samples)``.
            deg: Degraded stereo audio ``(batch, 2, samples)``.

        Returns:
            Dict with keys:

            - ``"fused"`` -- main quality score ``(batch,)`` in [0, 1].
            - ``"mon"`` -- monaural-only branch score ``(batch,)``.
            - ``"bin"`` -- binaural-only branch score ``(batch,)``.
        """
        ref_mon = self.mon_encoder(ref)
        deg_mon = self.mon_encoder(deg)
        ref_bin = self.bin_encoder(ref)
        deg_bin = self.bin_encoder(deg)

        if not isinstance(ref_mon, tuple):
            raise ValueError("Monaural encoder must return (left, right) tuple")

        ref_left, ref_right = ref_mon
        deg_left, deg_right = deg_mon

        ref_mid = 0.5 * (ref_left + ref_right)
        deg_mid = 0.5 * (deg_left + deg_right)
        delta_mon_mid = ref_mid - deg_mid

        delta_mon_features = self.mon_cnn(delta_mon_mid)
        delta_bin = ref_bin - deg_bin

        out_mon, _ = self.mon_lstm(delta_mon_features)
        pooled_mon = self.mon_norm(self.mon_pool(out_mon))

        out_bin, _ = self.bin_lstm(delta_bin)
        pooled_bin = self.bin_norm(self.bin_pool(out_bin))

        pooled_feats = self.pre_dense_norm(
            torch.cat([pooled_mon, pooled_bin], dim=-1)
        )

        return {
            "fused": torch.sigmoid(self.dense(pooled_feats)).squeeze(-1),
            "mon": torch.sigmoid(self.mon_head(pooled_mon)).squeeze(-1),
            "bin": torch.sigmoid(self.bin_head(pooled_bin)).squeeze(-1),
        }
