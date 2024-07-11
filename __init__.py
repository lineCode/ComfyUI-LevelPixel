from .modelunload_LP import ModelUnloader
from .hardmodelunload_LP import HardModelUnloader
from .softmodelunload_LP import SoftModelUnloader
from .llmloadersampler_LP import LLMOptionalMemoryFreeAdvanced

NODE_CLASS_MAPPINGS = {
"Model Unloader 🌸": ModelUnloader,
"Hard Model Unloader 🌸": HardModelUnloader,
"Soft Model Unloader 🌸": SoftModelUnloader,
"LLM Optional Memory Free Advanced 🌸": LLMOptionalMemoryFreeAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']