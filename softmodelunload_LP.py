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

class SoftModelUnloader:
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

    FUNCTION = "soft_unload_model"

    OUTPUT_NODE = True

    CATEGORY = "🌸LevelPixel/🌸Unloaders"

    def soft_unload_model(self, **kwargs):
        model_management.soft_empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        return (kwargs["source"],)

NODE_CLASS_MAPPINGS = {
    "SoftModelUnloader": SoftModelUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SoftModelUnloader": "Model Unloader 🌸",
}

