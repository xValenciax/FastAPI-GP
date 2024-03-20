from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from type_hints.model import TranslationOutput
import time
import os
import torch


class Translation_Model:

    __MODEL_NAME = "Selimx2001x/AraT5-Arabic-To-Sign-Language-Translation"
    __SAVE_PATH_TRANSLATION = r"models/Translation Model"

    def __init__(self) -> None:
        pass

    def __verify_model_path(self) -> bool:
        if len(os.listdir(os.path.join(os.getcwd(), self.__SAVE_PATH_TRANSLATION))) != 0:
            return True

        return False

    @property
    def model(self) -> dict[str, AutoModelForSeq2SeqLM | AutoTokenizer]:
        if self.__verify_model_path():
            my_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.__SAVE_PATH_TRANSLATION
            )

            my_tokenizer = AutoTokenizer.from_pretrained(
                self.__SAVE_PATH_TRANSLATION
            )

        else:
            my_model = AutoModelForSeq2SeqLM.from_pretrained(self.__MODEL_NAME)
            my_tokenizer = AutoTokenizer.from_pretrained(self.__MODEL_NAME)

        return {
            "model": my_model,
            "tokenizer": my_tokenizer
        }

    @classmethod
    def translate(self, _input: str) -> TranslationOutput:
        model, tokenizer = self().model.values()

        start_time = time.time()
        tokenized_input = tokenizer(
            _input, max_length=64, truncation=True, padding=True, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(**tokenized_input)

        # Decode the generated translations (if applicable)
        translated_text = tokenizer.batch_decode(
            output, skip_special_tokens=True)
        end_time = time.time()

        return TranslationOutput(translation=translated_text[0], time_taken=round((end_time - start_time) * 1000))

    @classmethod
    def save_model_files(self, model, tokenizer) -> None:
        model.save_pretrained(self.__SAVE_PATH_TRANSLATION)
        tokenizer.save_pretrained(self.__SAVE_PATH_TRANSLATION)
