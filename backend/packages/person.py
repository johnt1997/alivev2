from typing import Tuple, Union, List
import json
import os
import torch
from TTS.api import TTS
from playsound import playsound


class Person:
    def __init__(
        self,
        name: str,
        voice: str = "Male",
        personality: Union[List[Tuple[str, str]], None] = None,
    ):
        self.name = name
        self.personality = personality
        self.voice = voice  # str = self._get_voice_path()
        self._load_person()

    def _check_if_person_exists(self) -> bool:
        for root, dirs, files in os.walk("./persons/"):
            if self.name + ".json" in files:
                return True
        return False

    def _load_person(self) -> None:
        if self._check_if_person_exists() and self.personality is None:
            with open("./persons/" + self.name + ".json", "r") as json_file:
                data = json.load(json_file)
                personality_data = data.get("personality")
                self.personality = personality_data
            return
            # self.name = "Lelo"

        if self._check_if_person_exists() and self.personality is not None:
            # Read the JSON file
            with open("./persons/" + self.name + ".json", "r") as file:
                data = json.load(file)

            # Add the new field
            data["personality"] = self.personality

            # Save the modified JSON
            with open("./persons/" + self.name + ".json", "w") as file:
                json.dump(data, file, indent=2)

            return

        if self._check_if_person_exists() is False:
            data = {}
            data["name"] = self.name
            data["personality"] = self.personality

            with open("./persons/" + self.name + ".json", "w") as file:
                json.dump(data, file, indent=2)

            return

    def _get_voice_path(self) -> str:
        file_path = f"./voices/{self.name}/{self.name}.wav"
        if os.path.exists(file_path):
            return file_path
        else:
            return None

    def text_to_speach(self, text: str, index: int = 0, lang: str = "en") -> str:
        if self.voice == "Custom" and self._get_voice_path():
            tts = TTS(
                # model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                model_path="./models/tts_models--multilingual--multi-dataset--xtts_v2",
                config_path="./models/tts_models--multilingual--multi-dataset--xtts_v2/config.json",
                progress_bar=True,
                gpu=torch.cuda.is_available(),
            )  # .to(device)

            if not os.path.exists(f"./tts-outputs/{self.name}"):
                os.makedirs(f"./tts-outputs/{self.name}")

            tts.tts_to_file(
                text=text,
                speaker_wav=self._get_voice_path(),
                file_path=f"./tts-outputs/{self.name}/{lang + self.voice + index}.wav",
                language=lang,
            )

            return f"./tts-outputs/{self.name}/{lang + self.voice + index}.wav"
        else:
            tts = TTS(
                model_name="tts_models/en/vctk/vits",
                progress_bar=True,
                gpu=torch.cuda.is_available(),
            )  # .to(device)

            if not os.path.exists(f"./tts-outputs/{self.name}"):
                os.makedirs(f"./tts-outputs/{self.name}")

            if self.voice == "Male":
                speaker = "p256"
            if self.voice == "Female":
                speaker = "p225"

            tts.tts_to_file(
                text=text,
                file_path=f"./tts-outputs/{self.name}/{self.voice + str(index)}.wav",
                speaker=speaker,
            )

            return f"./tts-outputs/{self.name}/{self.voice + str(index)}.wav"

    def format_personality_prompt(self) -> str:
        index = 1
        formatted_personality = ""
        for orig_answer, reformulated_answer in self.personality:
            formatted_personality += f"Example #{index}\n"
            formatted_personality += f"Original Answer: {orig_answer}\n"
            formatted_personality += f"Reformulated Answer: {reformulated_answer}\n"
            formatted_personality += "\n"
            index += 1

        return formatted_personality
