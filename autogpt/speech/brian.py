import logging

import requests

from autogpt.speech.playback import play_audio
from autogpt.speech.base import VoiceBase


class BrianSpeech(VoiceBase):
    """Brian speech module for autogpt"""

    def _setup(self) -> None:
        """Setup the voices, API key, etc."""

    def _speech(self, text: str, _: int = 0) -> bool:
        """Speak text using Brian with the streamelements API

        Args:
            text (str): The text to speak

        Returns:
            bool: True if the request was successful, False otherwise
        """
        tts_url = (
            f"https://api.streamelements.com/kappa/v2/speech?voice=Brian&text={text}"
        )
        response = requests.get(tts_url)

        if response.status_code == 200:
            play_audio(response.content)
            return True
        else:
            logging.error(
                "Request failed with status code: %s, response content: %s",
                response.status_code,
                response.content,
            )
            return False
