from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict


app = FastAPI(
    title="NASA Space App Backend",
    description="Backend en FastAPI para análisis y predicción de exoplanetas 🚀",
    version="1.0.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

# --- Endpoint simple de salud ---
@app.get("/api/status")
def status():
    return {"status": "ok", "message": "Backend activo y listo para predicciones"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089, reload=True)


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8089, reload=False)