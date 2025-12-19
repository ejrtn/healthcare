from fastapi import APIRouter
from api.routes import(
    x_ray,
)

api_router = APIRouter()
api_router.include_router(x_ray.router)

@api_router.get("/")
def read_user_me():
    return "Index page"