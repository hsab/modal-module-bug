from modal import App, asgi_app
from .image import sdxl_image

app_fast_api = App("infra-fast-api")

with sdxl_image.imports():
    import fastapi
    from fastapi import Response
    from .sd import ModelSDXL


@app_fast_api.function(
    keep_warm=1,
    timeout=60 * 2,
    image=sdxl_image,
)
@asgi_app(label="fast-api")
def fast_api():
    web_app = fastapi.FastAPI()

    @web_app.get("/inference")
    async def inference(prompt: str, n_steps: int = 24, high_noise_frac: float = 0.8):
        return Response(
            content=ModelSDXL()
            .inference.remote(prompt, n_steps=n_steps, high_noise_frac=high_noise_frac)
            .getvalue(),
            media_type="image/jpeg",
        )

    return web_app
