# Copyright (c) vllm-GR Authors.
# Licensed under the Apache License, Version 2.0

from fastapi import FastAPI


def register_gr_pooling_api_routers(app: FastAPI) -> None:
    from vllm_gr.entrypoints.pooling.recommend.api_router import (
        router as recommend_router,
    )

    app.include_router(recommend_router)
