from torch import nn

from diffusers.models.transformers import CogVideoXTransformer3DModel


class CogVideoX(nn.Module):
    def __init__(
        self,
        from_pretrained,
        dtype,
        **kwargs,
    ):
        super().__init__()
        config = {"pretrained_model_name_or_path": from_pretrained, "torch_dtype": dtype}
        self.module = CogVideoXTransformer3DModel.from_pretrained(**config)