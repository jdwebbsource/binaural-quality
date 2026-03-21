"""Gammatone filterbank for auditory-inspired spectral decomposition."""

import torch
import torch.nn as nn
import torchaudio
import numpy as np


class GammatoneFilterBank(nn.Module):
    """ERB-spaced gammatone filterbank.

    Decomposes audio into frequency bands using gammatone filters spaced on
    the Equivalent Rectangular Bandwidth (ERB) scale, simulating the
    frequency selectivity of the human cochlea.

    Based on the Patterson-Holdsworth cochlear filter bank. Python
    implementation after Malcolm Slaney / Jason Heeris.

    Copyright (c) 1998, Malcolm Slaney <malcolm@interval.com>
    Copyright (c) 2009, Dan Ellis <dpwe@ee.columbia.edu>
    Copyright (c) 2014, Jason Heeris <jason.heeris@gmail.com>
    All rights reserved.

    The gammatone toolkit is licensed under the 3-clause
    BSD license: https://github.com/detly/gammatone/blob/master/COPYING

    Args:
        low_freq: Lowest centre frequency in Hz.
        sample_rate: Audio sample rate in Hz.
        filter_num: Number of gammatone filters.
    """

    def __init__(
        self,
        low_freq: float = 100.0,
        sample_rate: float = 32000,
        filter_num: int = 20,
    ):
        super().__init__()
        self.filter_num = filter_num
        self.low_freq = low_freq
        self.sample_rate = sample_rate
        self.high_freq = sample_rate / 2

        freqs = self._erb_space()
        coeffs = self._make_erb_filters(freqs)
        self.register_buffer("coeffs", coeffs)

    # ------------------------------------------------------------------
    # ERB scale helpers
    # ------------------------------------------------------------------

    def _erb_point(self, low_freq, high_freq, fraction):
        ear_q = 9.26449
        min_bw = 24.7
        return (
            -ear_q * min_bw
            + torch.exp(
                fraction
                * (
                    -torch.log(torch.tensor(high_freq) + ear_q * min_bw)
                    + torch.log(torch.tensor(low_freq) + ear_q * min_bw)
                )
            )
            * (torch.tensor(high_freq) + ear_q * min_bw)
        )

    def _erb_space(self):
        return self._erb_point(
            self.low_freq,
            self.high_freq,
            torch.arange(1, self.filter_num + 1) / self.filter_num,
        )

    # ------------------------------------------------------------------
    # Filter coefficient computation
    # ------------------------------------------------------------------

    def _make_erb_filters(self, centre_freqs, width=1.0):
        T = 1 / self.sample_rate
        ear_q = 9.26449
        min_bw = 24.7
        order = 1

        erb = width * ((centre_freqs / ear_q) ** order + min_bw**order) ** (
            1 / order
        )
        B = 1.019 * 2 * torch.pi * erb

        arg = (2 * centre_freqs * torch.pi * T).to(torch.complex64)
        vec = torch.exp(2j * arg)

        B0 = 1
        B1 = -2 * torch.cos(arg) / torch.exp(B * T)
        B2 = torch.exp(-2 * B * T)

        rt_pos = float(np.sqrt(3 + 2**1.5))
        rt_neg = float(np.sqrt(3 - 2**1.5))

        common = -T * torch.exp(-(B * T))

        k11 = torch.cos(arg) + rt_pos * torch.sin(arg)
        k12 = torch.cos(arg) - rt_pos * torch.sin(arg)
        k13 = torch.cos(arg) + rt_neg * torch.sin(arg)
        k14 = torch.cos(arg) - rt_neg * torch.sin(arg)

        A11 = common * k11
        A12 = common * k12
        A13 = common * k13
        A14 = common * k14

        gain_arg = torch.exp(1j * arg - B * T)
        gain = torch.abs(
            (vec - gain_arg * k11)
            * (vec - gain_arg * k12)
            * (vec - gain_arg * k13)
            * (vec - gain_arg * k14)
            * (
                T
                * torch.exp(B * T)
                / (-1 / torch.exp(B * T) + 1 + vec * (1 - torch.exp(B * T)))
            )
            ** 4
        )

        allfilts = torch.ones_like(centre_freqs)
        fcoefs = torch.column_stack(
            [
                T * allfilts, A11, A12, A13, A14, 0 * allfilts,
                B0 * allfilts, B1, B2,
                gain,
            ]
        )
        return fcoefs.real.to(torch.float32)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _erb_filterbank(self, x: torch.Tensor) -> torch.Tensor:
        bs, chs, seq_len = x.shape
        num_filters = self.coeffs.shape[0]

        gain = self.coeffs[:, 9]
        Bs = self.coeffs[:, 6:9]

        As_sets = [
            self.coeffs[:, (0, 1, 5)],
            self.coeffs[:, (0, 2, 5)],
            self.coeffs[:, (0, 3, 5)],
            self.coeffs[:, (0, 4, 5)],
        ]

        x_expanded = x.unsqueeze(2).expand(bs, chs, num_filters, seq_len)
        x_flat = x_expanded.reshape(-1, seq_len)

        Bs_flat = (
            Bs.unsqueeze(0).unsqueeze(0).expand(bs, chs, -1, -1).reshape(-1, 3)
        )
        gain_flat = (
            gain.unsqueeze(0).unsqueeze(0).expand(bs, chs, -1).reshape(-1)
        )

        y = x_flat
        for As in As_sets:
            As_flat = (
                As.unsqueeze(0).unsqueeze(0).expand(bs, chs, -1, -1).reshape(-1, 3)
            )
            y = torchaudio.functional.lfilter(y, Bs_flat, As_flat)

        y = y / gain_flat[:, None]
        return y.view(bs, chs, num_filters, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gammatone filterbank to input audio.

        Args:
            x: Audio tensor of shape ``(batch, channels, samples)``.

        Returns:
            Filtered output of shape ``(batch, channels, num_filters, samples)``.
        """
        return self._erb_filterbank(x)
