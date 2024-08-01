import modal

from .fastapi import app_fast_api
from .sd import app_sd

app = modal.App("multi-file-app")
app.include(app_fast_api)
app.include(app_sd)
