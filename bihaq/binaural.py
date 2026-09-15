"""Binaural spatial metrics: ILD, IPD, and IACC from stereo audio."""

import torch
import torch.nn as nn

from bihaq.dsp import GammatoneFilterBank


class BinauralMetrics(nn.Module):
    """Frame-wise binaural cues from stereo audio via gammatone filterbank.

    Extracts Interaural Level Difference (ILD), Interaural Phase
    Difference (IPD), and Interaural Cross-Correlation (IACC) in each
    gammatone frequency band.

    Args:
        frame_len: Analysis frame length in samples.
        hop_size: Hop size between frames in samples.
        sample_rate: Audio sample rate in Hz.
        filter_num: Number of gammatone filters.
    """

    def __init__(
        self,
        frame_len: int = 2048,
        hop_size: int = 1024,
        sample_rate: int = 44100,
        filter_num: int = 20,
    ):
        super().__init__()
        self.frame_len = frame_len
        self.hop_size = hop_size
        self.num_filt = filter_num
        self.gammatone = GammatoneFilterBank(
            sample_rate=sample_rate,
            filter_num=filter_num,
        )

    def _calculate_ild(self, x: torch.Tensor) -> torch.Tensor:
        l_rms = torch.sqrt(torch.mean(x[:, 0, :, :] ** 2, dim=-1))
        r_rms = torch.sqrt(torch.mean(x[:, 1, :, :] ** 2, dim=-1))
        return 20 * torch.log10((l_rms + 1e-12) / (r_rms + 1e-12))

    def _calculate_ipd(self, x: torch.Tensor) -> torch.Tensor:
        left = x[:, 0, :, :]
        right = x[:, 1, :, :]
        left_phase = torch.angle(torch.fft.rfft(left, dim=-1))
        right_phase = torch.angle(torch.fft.rfft(right, dim=-1))
        ipd_angle = left_phase - right_phase
        return torch.atan2(
            torch.mean(torch.sin(ipd_angle), dim=-1),
            torch.mean(torch.cos(ipd_angle), dim=-1),
        )

    def _calculate_iacc(self, x: torch.Tensor) -> torch.Tensor:
        left = x[:, 0, :, :]
        right = x[:, 1, :, :]
        left_fft = torch.fft.rfft(left, n=self.frame_len, dim=-1)
        right_fft = torch.fft.rfft(right, n=self.frame_len, dim=-1)
        cross_fft = left_fft * torch.conj(right_fft)
        corr = torch.fft.irfft(cross_fft, n=self.frame_len, dim=-1)
        cxx0 = torch.sum(left**2, dim=-1, keepdim=True)
        cyy0 = torch.sum(right**2, dim=-1, keepdim=True)
        corr = corr / (torch.sqrt(cxx0 * cyy0) + 1e-10)
        iacc, _ = torch.max(torch.abs(corr), dim=-1)
        return iacc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute binaural features from stereo audio.

        Args:
            x: Stereo audio tensor ``(batch, 2, samples)``.

        Returns:
            Binaural features ``(batch, 3 * filter_num, num_frames)`` --
            concatenated ILD, IPD, and IACC per gammatone band.
        """
        bs, _, seq_len = x.size()
        num_frames = (seq_len - self.frame_len) // self.hop_size + 1
        x_filt = self.gammatone(x)

        frames = []
        for i in range(num_frames):
            start = i * self.hop_size
            stop = start + self.frame_len
            x_frame = x_filt[:, :, :, start:stop]
            if x_frame.shape[-1] < self.frame_len:
                pad_amount = self.frame_len - x_frame.shape[-1]
                x_frame = nn.functional.pad(x_frame, (0, pad_amount))

            ild = self._calculate_ild(x_frame)
            ipd = self._calculate_ipd(x_frame)
            iacc = self._calculate_iacc(x_frame)
            frames.append(torch.cat([ild, ipd, iacc], dim=1))

        return torch.stack(frames, dim=-1)
