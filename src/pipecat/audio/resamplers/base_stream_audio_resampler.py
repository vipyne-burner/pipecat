#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
from abc import ABC, abstractmethod


class BaseStreamAudioResampler(ABC):
    """
    Abstract base class for streamable audio resampling.
    This interface assumes a stateful resampler that maintains context
    between successive calls to avoid artifacts like clicks at chunk boundaries.
    """

    def __init__(self, in_rate: float, out_rate: float):
        """
        Initializes the resampler with input and output sample rates.

        Parameters:
            in_rate (float): Input sample rate in Hz.
            out_rate (float): Output sample rate in Hz.
        """
        self.in_rate = in_rate
        self.out_rate = out_rate

    @abstractmethod
    async def resample(self, audio: bytes) -> bytes:
        """
        Resample a chunk of audio data.

        Parameters:
            audio (bytes): Audio data as a byte string.

        Returns:
            bytes: Resampled audio data as a byte string.
        """
        pass
