from fastapi import APIRouter
from type_hints.model import TranslationInput, ModelName
from Services.Translate.model import Translation_Model
from Services.Speech.model import Speech_to_Text_Model


router = APIRouter()


@router.get('/')
async def home():
    return {"Message": "Welcome To Our API, Developed By Selim using FastAPI.", }


@router.post("/models/{model_name}")
async def get_translation(model_name: ModelName, _input: TranslationInput):
    output = None
    if model_name == ModelName.Translation:
        output = Translation_Model.translate(_input.text)

    elif model_name == ModelName.Speech:
        output = Speech_to_Text_Model()

    return output
