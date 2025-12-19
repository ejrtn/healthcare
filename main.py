from fastapi import FastAPI, Form, Request
from api.main import api_router
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 실행 명령어 : uvicorn main:app --reload

app = FastAPI()

templates = Jinja2Templates(directory="templates")
@app.get("/x-ray", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("x-ray.html", {"request": request})

app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 곳에서 접속 허용 (개발용)
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)