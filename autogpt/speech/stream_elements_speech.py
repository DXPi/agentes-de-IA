from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import requests
from playsound import playsound

if TYPE_CHECKING:
    from autogpt.config import Config

from autogpt.speech.base import VoiceBase

logger = logging.getLogger(__name__)


class StreamElementsSpeech(VoiceBase):
    """Streamelements speech module for autogpt"""

    def _setup(self, config: Config) -> None:
        """Setup the voices, API key, etc."""

    def _speech(self, text: str, voice: str, _: int = 0) -> bool:
        voice1 = os.getenv("STREAMELEMENTS_VOICE_1") or "Brian"
        voice2 = os.getenv("STREAMELEMENTS_VOICE_2") or "Emma"
        voice3 = os.getenv("STREAMELEMENTS_VOICE_3") or "Amy"

        """Speak text using the streamelements API

        Args:
            text (str): The text to speak
            voice (str): The voice to use

        Returns:
            bool: True if the request was successful, False otherwise
        """
        if voice == 2:
            tts_url = f"https://api.streamelements.com/kappa/v2/speech?voice={voice3}&text={text}"
        elif voice == 1:
            tts_url = f"https://api.streamelements.com/kappa/v2/speech?voice={voice2}&text={text}"
        else:
            tts_url = f"https://api.streamelements.com/kappa/v2/speech?voice={voice1}&text={text}"
        response = requests.get(tts_url)

        if response.status_code == 200:
            with open("speech.mp3", "wb") as f:
                f.write(response.content)
            playsound("speech.mp3")
            os.remove("speech.mp3")
            return True
        else:
            logger.error(
                "Request failed with status code: %s, response content: %s",
                response.status_code,
                response.content,
            )
            return False
