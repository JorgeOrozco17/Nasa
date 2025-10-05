from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import songs
from routers import graph

app = FastAPI(title="FastAPI Template Chavos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(graph.router, prefix="/api")
app.include_router(songs.router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8089, reload=False)