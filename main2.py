import os
import cv2
import math
import fitz
import easyocr
from cv2 import Mat
from numpy import ndarray
from PyPDF2 import PdfReader
from typing import Tuple, Any
from img2table.ocr import EasyOCR
from pdf2image import convert_from_path
from img2table.document import Image, PDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal


def convert_pdf_to_image(pdf_path: str) -> Tuple[str, Tuple[int, int]]:
    """
    Конвертация первой страницы PDF в изображение и сохранение его как JPG.
    Возвращает путь к сохраненному изображению.
    """
    images = convert_from_path(pdf_path)
    image_path = f'{os.path.splitext(pdf_path)[0]}.jpg'
    images[0].save(image_path, 'JPEG')
    image_size = images[0].size  # (width, height)
    return image_path, image_size


def get_pdf_page_size(pdf_path: str) -> Tuple[int, int]:
    """
    Получает размеры страницы PDF в точках (points).
    Возвращает ширину и высоту страницы.
    """
    reader = PdfReader(pdf_path)
    first_page = reader.pages[0]
    width = int(first_page.mediabox.width)
    height = int(first_page.mediabox.height)
    return width, height


def get_text_coordinates(image_path: str, lang_selected: list) \
        -> tuple[list[tuple[tuple[int, int, int, int], Any]], Mat | ndarray]:
    """
    Извлечение координат текста из изображения с использованием easyocr.
    Возвращает список координат и распознанного текста.
    """
    reader = easyocr.Reader(lang_selected)
    image = cv2.imread(image_path, 0)
    contours = reader.readtext(image, paragraph=True, x_ths=1.0, y_ths=0.6)
    text_coordinates = []

    for box, text in contours:
        # Получаем координаты
        top_left, top_right, bottom_right, bottom_left = box
        x1, y1 = int(top_left[0]), int(top_left[1])
        x2, y2 = int(bottom_right[0]), int(bottom_right[1])
        text_coordinates.append(((x1, y1, x2, y2), text))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)
    cv2.imwrite(f"{image_path}_rect.jpg", image)

    return text_coordinates, image


def adjust_coordinates(
        coordinates: Tuple[int, int, int, int],
        img_size: Tuple[int, int],
        pdf_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Преобразует координаты из изображения в координаты на PDF с учетом масштабирования.
    """
    img_width, img_height = img_size
    pdf_width, pdf_height = pdf_size

    x1, y1, x2, y2 = coordinates

    # Рассчитываем коэффициенты масштабирования
    scale_x = pdf_width / img_width
    scale_y = pdf_height / img_height

    # Преобразуем координаты
    x1_pdf = x1 * scale_x
    y1_pdf = (img_height - y1) * scale_y  # переворачиваем по Y
    x2_pdf = x2 * scale_x
    y2_pdf = (img_height - y2) * scale_y  # переворачиваем по Y

    return int(x1_pdf), int(y2_pdf), int(x2_pdf), int(y1_pdf)


def adjust_for_spaces(line_text: str, line_x1: int, line_x2: int, space_width: int = 4) -> Tuple[int, int]:
    """
    Корректировка координат `x1` и `x2` на основе пробелов в начале и конце строки.
    """
    # Удаляем символы новой строки
    stripped_text = line_text.rstrip('\n')

    # Количество пробелов в начале и конце строки
    num_leading_spaces = len(stripped_text) - len(stripped_text.lstrip(' '))
    num_trailing_spaces = len(stripped_text) - len(stripped_text.rstrip(' '))

    # Корректировка `x1` (уменьшаем x1) и `x2` (увеличиваем x2)
    line_x1_adjusted = line_x1 + num_leading_spaces * space_width
    line_x2_adjusted = line_x2 - num_trailing_spaces * space_width

    return line_x1_adjusted, line_x2_adjusted


def extract_text_within_coordinates(pdf_path: str, coordinates: Tuple[int, int, int, int]) -> str:
    """
    Извлечение текста из PDF внутри заданных координат с использованием pdfminer.
    """
    x1, y1, x2, y2 = coordinates
    extracted_text = ""

    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for text_line in element:
                    # Округляем координаты text_line до ближайшего меньшего целого числа
                    line_x1 = math.floor(text_line.bbox[0])
                    line_y1 = math.floor(text_line.bbox[1])
                    line_x2 = math.floor(text_line.bbox[2])
                    line_y2 = math.floor(text_line.bbox[3])

                    line_text = text_line.get_text()

                    # Корректируем `x1` и `x2` на основе пробелов
                    line_x1_adjusted, line_x2_adjusted = adjust_for_spaces(line_text, line_x1, line_x2)

                    # Проверка координат с учетом коррекции
                    if x1 <= line_x1_adjusted + offset \
                            and y1 <= line_y1 + offset \
                            and x2 + offset >= line_x2_adjusted \
                            and y2 + offset >= line_y2:
                        extracted_text += f"{line_text.strip()} "

    return extracted_text.strip()


def is_pdf_image_based(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():  # Если на странице есть текст, то файл не является чистым изображением
            return False
    return True  # Если на всех страницах нет текста, то PDF вероятно содержит изображения


def process_pdf(file_path: str, pdf, lang_selected: list):
    """
    Основная функция обработки PDF-файла: получение координат текста через easyocr,
    затем чтение текста из PDF по этим координатам через pdfminer.
    """
    # Конечный текст из pdf
    text = ""
    # Конвертируем первую страницу PDF в изображение
    is_img = is_pdf_image_based(pdf)
    # Получаем координаты текста с использованием easyocr
    text_coordinates, image = get_text_coordinates(file_path, lang_selected)

    # Извлекаем текст из PDF на основе координат
    for coords, ocr_text in text_coordinates:
        # Преобразуем координаты в систему координат PDF
        # pdf_coords = adjust_coordinates(coords, img_size, pdf_size)
        # extracted_text = extract_text_within_coordinates(file_path, coords)
        print(f"Координаты блока: {coords}")
        print(f"OCR текст: {ocr_text}")
        print("-" * 40)
        text += ocr_text + "\n\n"
    print(text)
    return image, text


def process_img2table_jpg(file_path: str, checkbox, lang_selected: list):
    # Instantiation of OCR
    ocr = EasyOCR(lang=lang_selected)
    # Instantiation of document, either an image or a PDF
    doc = Image(file_path)
    # Table extraction
    extracted_tables = doc.extract_tables(
        ocr=ocr,
        implicit_rows=False,
        borderless_tables=checkbox,
        min_confidence=50
    )
    doc.to_xlsx(
        dest=f"{os.path.basename(file_path)}.xlsx",
        ocr=ocr,
        implicit_rows=False,
        borderless_tables=True,
        min_confidence=50
    )
    text = ""
    for elem in extracted_tables:
        # Получаем координаты
        for i in elem.content:
            for cell in elem.content[i]:
                if cell.value:
                    text += cell.value + "\n"
                cv2.rectangle(
                    doc.images[0],
                    [cell.bbox.x1, cell.bbox.y1],
                    [cell.bbox.x2, cell.bbox.y2],
                    (0, 0, 0),
                    2
                )
    print(text)
    return doc.images[0], text


def process_img2table_pdf(file_path: str, lang_selected: list):
    pdf = PDF(
        file_path,
        detect_rotation=True,
        pdf_text_extraction=True
    )
    if is_pdf_image_based(file_path):
        ocr = EasyOCR(lang=lang_selected)
        extracted_tables = pdf.extract_tables(
            ocr=ocr,
            implicit_rows=False,
            borderless_tables=False,
            min_confidence=50
        )
    else:
        ocr = None
        extracted_tables = pdf.extract_tables(
            ocr=ocr,
            implicit_rows=False,
            borderless_tables=False
        )
    pdf.to_xlsx(
        dest=f"{os.path.basename(file_path)}.xlsx",
        ocr=ocr,
        implicit_rows=False,
        borderless_tables=False,
        min_confidence=50
    )
    text = ""
    for elems in extracted_tables.values():
        # Получаем координаты
        for elem in elems:
            for i in elem.content:
                for cell in elem.content[i]:
                    if cell.value:
                        text += cell.value + "\n"
                    cv2.rectangle(
                        pdf.images[0],
                        [cell.bbox.x1, cell.bbox.y1],
                        [cell.bbox.x2, cell.bbox.y2],
                        (0, 0, 0),
                        2
                    )
    cv2.imwrite(f"{file_path}_rect.jpg", pdf.images[0])
    return pdf.images[0], text


if __name__ == "__main__":
    # Запуск процесса для указанного PDF-файла
    offset = 8
    process_pdf("/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page0.pdf")
