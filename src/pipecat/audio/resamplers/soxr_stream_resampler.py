#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time

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
        self.last_resample_time = time.time()
        self.CLEAR_AFTER_SECS = .2  # Clear state after 200ms of inactivity
        self.soxr_stream = soxr.ResampleStream(
            in_rate=in_rate, out_rate=out_rate, num_channels=1, quality="VHQ", dtype="int16"
        )

    def _maybe_clear_internal_state(self):
        current_time = time.time()
        time_since_last_resample = (current_time - self.last_resample_time)
        # If more than CLEAR_AFTER_MS milliseconds have passed, clear the resampler state
        if time_since_last_resample > self.CLEAR_AFTER_SECS:
            self.soxr_stream.clear()
        self.last_resample_time = current_time

    async def resample(self, audio: bytes) -> bytes:
        if self.in_rate == self.out_rate:
            return audio

        self._maybe_clear_internal_state()

        audio_data = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = self.soxr_stream.resample_chunk(audio_data)
        result = resampled_audio.astype(np.int16).tobytes()
        return result
