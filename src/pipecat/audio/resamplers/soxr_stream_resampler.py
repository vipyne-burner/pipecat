#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
import soxr

from pipecat.audio.resamplers.base_stream_audio_resampler import BaseStreamAudioResampler


class SOXRStreamAudioResampler(BaseStreamAudioResampler):
    """
    Audio resampler implementation using the SoX resampler library.
    It keeps an internal history which avoids clicks at chunk boundaries
    """

    def __init__(self, in_rate: float, out_rate: float):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.soxr_stream = soxr.ResampleStream(
            in_rate=in_rate, out_rate=out_rate, num_channels=1, quality="VHQ", dtype="int16"
        )

    async def resample(self, audio: bytes) -> bytes:
        if self.in_rate == self.out_rate:
            return audio
        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = self.soxr_stream.resample_chunk(audio_data)
        result = resampled_audio.astype(np.int16).tobytes()
        return result
