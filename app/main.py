import uvicorn
from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(title="Agent API Service")

# 引入路由
app.include_router(api_router, prefix="/app/v1")

@app.get("/")
async def root():
    return {"message": "Agent service is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
