from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from type_hints.model import TranslationOutput
import os


class Speech_to_Text_Model:
    __MODEL_NAME = "Selimx2001x/AraT5-Arabic-To-Sign-Language-Translation"
    __SAVE_PATH_ASR = "models/Translation Model"

    def __init__(self) -> None:
        pass

    def __verify_model_path(self) -> bool:
        if len(os.listdir(os.path.join(os.getcwd(), "..", "..", self.__SAVE_PATH_ASR))) != 0:
            return True

        return False

    @property
    def model(self) -> dict:
        if self.__verify_model_path():
            my_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.__SAVE_PATH_ASR
            )

            my_tokenizer = AutoTokenizer.from_pretrained(
                self.__SAVE_PATH_ASR
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
        model, tokenizer = self.model.values()

    @classmethod
    def save_model_files(self, model, tokenizer) -> None:
        model.save_pretrained(self.__SAVE_PATH_TRANSLATION)
        tokenizer.save_pretrained(self.__SAVE_PATH_TRANSLATION)
