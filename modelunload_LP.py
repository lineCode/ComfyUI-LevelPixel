from comfy import model_management
from llama_cpp import Llama
import gc
import torch

# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
any = AnyType("*")

class ModelUnloader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
         return {
            "required": {
                "source": (any, {}),
            },
        }

    RETURN_TYPES = (any,)

    FUNCTION = "unload_model"

    OUTPUT_NODE = True

    CATEGORY = "🌸LevelPixel/🌸Unloaders"

    def unload_model(self, **kwargs):
        loadedmodels=model_management.current_loaded_models
        unloaded_model = False
        for i in range(len(loadedmodels) -1, -1, -1):
            m = loadedmodels.pop(i)
            m.model_unload()
            del m
            unloaded_model = True
        if unloaded_model:
            model_management.soft_empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        return (kwargs["source"],)

NODE_CLASS_MAPPINGS = {
    "ModelUnloader": ModelUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelUnloader": "Model Unloader 🌸",
}

