from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from .routes import foo, add_document, jurisprudence, expansion


app = FastAPI()

app.include_router(add_document.router)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, foo.chain, disabled_endpoints=["batch"], path="/foo")
add_routes(app, jurisprudence.qa_chain, disabled_endpoints=["batch"], path="/jurisprudence")
add_routes(app, expansion.expansion_chain, disabled_endpoints=["batch"], path="/expansion")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
