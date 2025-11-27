# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List, Union, Tuple
import inspect

import torch

from diffusers.models.embeddings import get_3d_rotary_pos_embed

from mindspeed_mm.tasks.inference.pipeline.pipeline_base import MMPipeline
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


class CogVideoXPipeline(MMPipeline, InputsCheckMixin, MMEncoderMixin):
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(self, vae, text_encoder, tokenizer, scheduler, predict_model, config=None):
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae.module,
            predict_model=predict_model.module, scheduler=scheduler
        )

        self.vae = vae.module
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.predict_model = predict_model.module

        config = config.to_dict()
        self.num_frames, self.height, self.width = config.get("input_size", [49, 480, 720])
        self.generator = torch.Generator().manual_seed(config.get("seed", 42))
        self.num_videos_per_prompt = 1
        self.max_sequence_length = 226

        self.scheduler.use_dynamic_cfg = config.get("use_dynamic_cfg", True)

        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.predict_model.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.predict_model.config.patch_size)
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.predict_model.config.patch_size)
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.predict_model.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.predict_model.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            use_real=True,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @torch.no_grad()
    def __call__(self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        if self.num_frames > 49:
            raise ValueError(
                "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
            )

        height = self.height or self.predict_model.config.sample_size * self.vae_scale_factor_spatial
        width = self.width or self.predict_model.config.sample_size * self.vae_scale_factor_spatial

        # 1. Check inputs.
        self.text_prompt_checks(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self.generate_params_checks(height, width)
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.text_encoder.device or self._execution_device

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.encode_texts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=device,
            do_classifier_free_guidance=True,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_length=self.max_sequence_length,
            clean_caption=False,
            prompt_to_lower=False
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 5. Prepare latents
        latent_channels = self.predict_model.config.in_channels
        batch_size = batch_size * self.num_videos_per_prompt
        shape = (
            batch_size,
            (self.num_frames - 1) // self.vae_scale_factor_temporal + 1,
            latent_channels,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial
        )
        latents = self.prepare_latents(shape, generator=self.generator, device=device, dtype=prompt_embeds.dtype,
                                       latents=latents)
        # 6 prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.predict_model.config.use_rotary_positional_embeddings
            else None
        )

        model_kwargs = {"encoder_hidden_states": prompt_embeds,
                        "image_rotary_emb": image_rotary_emb,
                        "return_dict": False,
                        "hidden_states": True}

        latents = self.scheduler.sample(model=self.predict_model, shape=shape, latents=latents,
                                        model_kwargs=model_kwargs,
                                        extra_step_kwargs=extra_step_kwargs)

        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents
        video = self.decode_latents(latents).sample
        return video

    def callback_on_step_end_tensor_inputs_checks(self, callback_on_step_end_tensor_inputs):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
