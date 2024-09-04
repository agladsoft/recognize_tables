import os
import logging
from typing import List
from app import process_pdf, process_image, get_engines
from fastapi import APIRouter, Request, UploadFile, File, Depends, Query

recognize_router = APIRouter()
FILES_DIR: str = "."


class QueryParams:
    def __init__(
        self,
        is_table_bordered: bool = Query(False, description="Если у таблицы есть структура, установите галочку"),
        selected_engine: str = Query(get_engines()[1], enum=get_engines(), description="Выберите движок"),
        selected_languages: List[str] = Query(
            ["eng"],
            enum=["eng", "rus", "ch_sim", "ch_tra", "ara"],
            description="Выберите языки"
        ),
        only_ocr: bool = Query(True, description="Простое распознавание текста"),
        confidence: int = Query(50, description="Уверенность распознанного текста"),
        x_shift: float = Query(1.0, description="Сдвиг по X для EasyOCR"),
        y_shift: float = Query(0.6, description="Сдвиг по Y для EasyOCR"),
        psm: int = Query(11, description="PSM для Tesseract")
    ):
        self.is_table_bordered = is_table_bordered
        self.selected_engine = selected_engine
        self.selected_languages = selected_languages
        self.only_ocr = only_ocr
        self.confidence = confidence
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.psm = psm


def save_files(file: UploadFile) -> str:
    if file.filename is None or file.filename == "":
        raise FileNotFoundError("Файл не был загружен")
    file_location = f"{FILES_DIR}/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return file_location


@recognize_router.post("/recognize", tags=["Recognize"])
def get_text_from_image(request: Request, file: UploadFile = File(...), params: QueryParams = Depends()):
    logging.info(f"Files {file}")
    dict_form = request._form._dict
    print(dict_form)
    file_path = save_files(file)
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        obj = process_pdf(
            pdf_file=file_path,
            is_table_bordered=params.is_table_bordered,
            selected_engine=params.selected_engine,
            selected_languages=params.selected_languages,
            only_ocr=params.only_ocr,
            confidence=params.confidence,
            x_shift=params.x_shift,
            y_shift=params.y_shift,
            psm=params.psm
        )
        return obj[1]
    elif ext in [".jpg", ".jpeg", ".png"]:
        obj = process_image(
            image_data={"background": file_path},
            is_table_bordered=params.is_table_bordered,
            selected_engine=params.selected_engine,
            selected_languages=params.selected_languages,
            only_ocr=params.only_ocr,
            confidence=params.confidence,
            x_shift=params.x_shift,
            y_shift=params.y_shift,
            psm=params.psm
        )
    else:
        raise FileNotFoundError("Файл не является изображением или PDF")
    return obj[1]
