import io
from pathlib import Path
import os
import gc

from modal import App, build, enter, gpu, method, web_endpoint, Volume
from .image import sdxl_image


app_sd = App("infra-sd-xl")
vol_models = Volume.from_name("models")
vol_loras = Volume.from_name("loras")
vol_hfcache = Volume.from_name("hfcache")


with sdxl_image.imports():
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/hfcache/"
    os.environ["TRANSFORMERS_CACHE"] = "/root/hfcache/"
    os.environ["HF_HOME"] = "/root/hfcache/"

    import torch
    from diffusers import DiffusionPipeline
    from fastapi import Response

load_options = dict(
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    device_map="auto",
)


@app_sd.cls(
    gpu=gpu.A10G(),
    container_idle_timeout=240,
    image=sdxl_image,
    volumes={
        "/root/models": vol_models,
        "/root/loras": vol_loras,
        "/root/hfcache": vol_hfcache,
    },
)
class ModelSDXL:
    def download_models(sef, model: str):
        from huggingface_hub import snapshot_download

        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]

        snapshot_download(
            model,
            ignore_patterns=ignore,
        )
        vol_hfcache.commit()

    @build()
    def build(self):
        self.download_models("stabilityai/stable-diffusion-xl-base-1.0")
        self.download_models("stabilityai/stable-diffusion-xl-refiner-1.0")

    @enter()
    def enter(self):
        if hasattr(self, "base"):
            del self.base
        if hasattr(self, "refiner"):
            del self.refiner

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Load base model
        # if not hasattr(self, "base"):
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Load refiner model
        # if not hasattr(self, "refiner"):
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

    def _inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        negative_prompt = "disfigured, ugly, deformed"

        image = self.base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        image = self.refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")

        return byte_stream

    @method()
    def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
        return self._inference(
            prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
        ).getvalue()

    @web_endpoint(docs=True)
    def web_inference(
        self, prompt: str, n_steps: int = 24, high_noise_frac: float = 0.8
    ):
        return Response(
            content=self._inference(
                prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
            ).getvalue(),
            media_type="image/jpeg",
        )
