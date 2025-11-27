# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
# Copyright 2024 Stability AI and The HuggingFace Team

import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                torch_dtype=torch.float16, local_files_only=True)
pipe = pipe.to("npu")

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]
image.save("sd3_hello_world.png")
