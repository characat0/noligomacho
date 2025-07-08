from fastapi import APIRouter, Depends, HTTPException, UploadFile
from app.services.vector_store import VectorStoreService

router = APIRouter(
    prefix="/vector-store",
)

@router.post("/add-document", status_code=201)
def add_document(
        files: list[UploadFile],
        vector_store: VectorStoreService = Depends(VectorStoreService),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    fs = [
        x.file for x in files
    ]
    ids = vector_store.add_files(fs)
    return {
        "ids": ids,
    }
