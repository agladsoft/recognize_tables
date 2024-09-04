import os
import logging
from typing import List, Union
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from app import process_pdf, process_image, get_engines, get_supported_languages

recognize_router = APIRouter()
FILES_DIR: str = "."


def validate_languages(selected_languages: Union[str, List[str]], selected_engine: str):
    # Получаем поддерживаемые языки для выбранного движка
    supported_languages = get_supported_languages(selected_engine)

    if selected_engine in {"EasyOCR", "TesseractOCR"}:
        # Для EasyOCR и TesseractOCR должны быть списки
        if isinstance(selected_languages, str):
            selected_languages = [selected_languages]
        for lang in selected_languages:
            if lang not in supported_languages:
                raise HTTPException(
                    status_code=400,
                    detail=f"Язык '{lang}' не поддерживается движком '{selected_engine}'. "
                           f"Поддерживаемые языки: {supported_languages}"
                )
    elif selected_engine == "PaddleOCR":
        # Для PaddleOCR должна быть строка
        if isinstance(selected_languages, list):
            if len(selected_languages) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Для движка '{selected_engine}' можно передать только один язык в виде строки."
                )
            selected_languages = selected_languages[0]  # Преобразуем список в строку
        if selected_languages not in supported_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Язык '{selected_languages}' не поддерживается движком '{selected_engine}'. "
                       f"Поддерживаемые языки: {supported_languages}"
            )

    return selected_languages


def save_files(file: UploadFile) -> str:
    if file.filename is None or file.filename == "":
        raise FileNotFoundError("Файл не был загружен")
    file_location = os.path.join(FILES_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    return file_location


@recognize_router.post("/recognize", tags=["Recognize"])
def get_text_from_image(
        file: UploadFile = File(..., description="Загрузите файл изображения или PDF"),
        is_table_bordered: bool = Query(False, description="Если у таблицы есть структура, установите галочку"),
        selected_engine: str = Query(get_engines()[1], enum=get_engines(), description="Выберите движок"),
        selected_languages: List[str] = Query(["eng"], description="Выберите языки"),
        only_ocr: bool = Query(True, description="Простое распознавание текста"),
        confidence: int = Query(50, description="Уверенность распознанного текста"),
        x_shift: float = Query(1.0, description="Сдвиг по X для EasyOCR"),
        y_shift: float = Query(0.6, description="Сдвиг по Y для EasyOCR"),
        psm: int = Query(11, description="PSM для Tesseract")
):
    logging.info(f"Полученный файл: {file.filename}")

    # Валидация языков на основе выбранного движка
    selected_languages = validate_languages(selected_languages, selected_engine)

    # Сохранение загруженного файла
    file_path = save_files(file)
    ext = os.path.splitext(file_path)[-1].lower()

    # Обработка PDF-файлов
    if ext == ".pdf":
        result = process_pdf(
            pdf_file=file_path,
            is_table_bordered=is_table_bordered,
            selected_engine=selected_engine,
            selected_languages=selected_languages,
            only_ocr=only_ocr,
            confidence=confidence,
            x_shift=x_shift,
            y_shift=y_shift,
            psm=psm
        )
        return result[1]

    # Обработка изображений
    elif ext in [".jpg", ".jpeg", ".png"]:
        result = process_image(
            image_data={"background": file_path},
            is_table_bordered=is_table_bordered,
            selected_engine=selected_engine,
            selected_languages=selected_languages,
            only_ocr=only_ocr,
            confidence=confidence,
            x_shift=x_shift,
            y_shift=y_shift,
            psm=psm
        )
        return result[1]

    else:
        raise FileNotFoundError("Файл не является изображением или PDF")

