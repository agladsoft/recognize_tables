import os
import cv2
import math
import pymupdf
import numpy as np
import pandas as pd
from cv2 import Mat
from numpy import ndarray
from PyPDF2 import PdfReader
from pandas import DataFrame
from pdf2image import convert_from_path
from img2table.document import Image, PDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal
from typing import Tuple, Any, List, Optional, Union
from img2table.ocr import EasyOCR, TesseractOCR, PaddleOCR, DocTR


class OCRBase:
    def __init__(self, lang: Union[List[str], str] = 'en'):
        self.lang = lang

    def perform_ocr(self, file_path: str) -> Tuple[Mat, str]:
        raise NotImplementedError("OCR method not implemented!")


class CoordinateAdjuster:
    offset = 5  # Смещение для точности сопоставления координат

    @staticmethod
    def adjust_coordinates(
            coordinates: Tuple[int, int, int, int],
            img_size: Tuple[int, int],
            pdf_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Преобразует координаты из изображения в координаты на PDF с учетом масштабирования.
        :param coordinates: Координаты текста (x1, y1, x2, y2).
        :param img_size: Размеры изображения (ширина, высота).
        :param pdf_size: Размеры PDF страницы (ширина, высота).
        :return: Преобразованные координаты.
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

    @staticmethod
    def adjust_for_spaces(line_text: str, line_x1: int, line_x2: int, space_width: int = 4) -> Tuple[int, int]:
        """
        Корректировка координат `x1` и `x2` на основе пробелов в начале и конце строки.
        :param line_text: Текст строки.
        :param line_x1: Начальная координата x1.
        :param line_x2: Конечная координата x2.
        :param space_width: Ширина пробела.
        :return:
        """
        stripped_text = line_text.rstrip('\n')

        # Количество пробелов в начале и конце строки
        num_leading_spaces = len(stripped_text) - len(stripped_text.lstrip(' '))
        num_trailing_spaces = len(stripped_text) - len(stripped_text.rstrip(' '))

        # Корректировка `x1` (уменьшаем x1) и `x2` (увеличиваем x2)
        line_x1_adjusted = line_x1 + num_leading_spaces * space_width
        line_x2_adjusted = line_x2 - num_trailing_spaces * space_width

        return line_x1_adjusted, line_x2_adjusted


class EasyOCREngine(OCRBase):
    def __init__(self, lang: Union[List[str], str] = 'en', x_shift: float = 1.0, y_shift: float = 0.6):
        super().__init__(lang)
        import easyocr
        self.ocr = easyocr.Reader(lang)
        self.x_shift = x_shift
        self.y_shift = y_shift

    @staticmethod
    def convert_pdf_to_image(file_path: str) -> Tuple[str, Tuple[int, int]]:
        """
        Конвертация первой страницы PDF в изображение и сохранение его как JPG.
        :return: Расположение изображения и его размеры.
        """
        images = convert_from_path(file_path)
        image_path = f'{os.path.splitext(file_path)[0]}.jpg'
        images[0].save(image_path, 'JPEG')
        image_size = images[0].size  # (width, height)
        return image_path, image_size

    @staticmethod
    def get_pdf_page_size(file_path: str, page: int) -> Tuple[int, int]:
        """
        Получает размеры страницы PDF в точках (points).
        :return: Ширина и высота страницы.
        """
        reader = PdfReader(file_path)
        first_page = reader.pages[page]
        width = int(first_page.mediabox.width)
        height = int(first_page.mediabox.height)
        return width, height

    @staticmethod
    def extract_text_within_coordinates(file_path: str, page_img: int, coordinates: Tuple[int, int, int, int]) -> str:
        """
        Извлечение текста из PDF внутри заданных координат с использованием pdfminer.
        :param file_path: Путь к файлу.
        :param page_img: Номер страницы изображения.
        :param coordinates: Координаты текста (x1, y1, x2, y2).
        :return: Извлеченный текст.
        """
        x1, y1, x2, y2 = coordinates
        extracted_text = ""

        for page_pdf, page_layout in enumerate(extract_pages(file_path)):
            if page_pdf != page_img:
                continue

            for element in page_layout:
                if isinstance(element, LTTextBoxHorizontal):
                    for text_line in element:
                        line_x1 = math.floor(text_line.bbox[0])
                        line_y1 = math.floor(text_line.bbox[1])
                        line_x2 = math.floor(text_line.bbox[2])
                        line_y2 = math.floor(text_line.bbox[3])

                        line_text = text_line.get_text()

                        # Корректируем `x1` и `x2` на основе пробелов
                        line_x1_adjusted, line_x2_adjusted = CoordinateAdjuster.adjust_for_spaces(
                            line_text, line_x1, line_x2
                        )

                        # Проверка координат с учетом коррекции
                        if x1 <= line_x1_adjusted + CoordinateAdjuster.offset \
                                and y1 <= line_y1 + CoordinateAdjuster.offset \
                                and x2 + CoordinateAdjuster.offset >= line_x2_adjusted \
                                and y2 + CoordinateAdjuster.offset >= line_y2:
                            extracted_text += f"{line_text.strip()} "

        return extracted_text.strip()

    def get_text_coordinates(self, image_path: str) \
            -> Tuple[List[Tuple[Tuple[int, int, int, int], Any]], Mat | ndarray]:
        """
        Извлечение координат текста из изображения с использованием easyocr.
        :param image_path: Путь к изображению.
        :return: Список координат и распознанного текста, изображение с выделенными прямоугольниками.
        """
        image = cv2.imread(image_path, 0)
        contours = self.ocr.readtext(image, paragraph=True, x_ths=self.x_shift, y_ths=self.y_shift)
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
    
    def get_text_from_image(self, file_path, page, convert_coord, img_size, pdf_size, text, text_coordinates) -> str:
        """
        Получение текста из изображения.
        :param file_path: Путь к изображению.
        :param page: Номер страницы изображения.
        :param convert_coord: Флаг конвертации координат.
        :param img_size: Размеры изображения.
        :param pdf_size: Размеры PDF страницы.
        :param text: Текст.
        :param text_coordinates: Список координат и распознанного текста.
        :return: Текст.
        """
        # Извлекаем текст
        for coord, ocr_text in text_coordinates:
            if convert_coord:
                pdf_coord = CoordinateAdjuster.adjust_coordinates(coord, img_size, pdf_size)
                text += self.extract_text_within_coordinates(file_path, page, pdf_coord) + "\n"
            else:
                text += ocr_text + "\n"
        return text

    def perform_ocr(self, file_path) -> Tuple[Mat | ndarray, str]:
        """
        Выполнение OCR с помощью EasyOCR.
        :param file_path: Путь к изображению.
        :return: Изображение с выделенными прямоугольниками, распознанный текст.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        is_pdf = ext == ".pdf"
        images: List[Image.images] | List[ndarray] = convert_from_path(file_path) if is_pdf else [cv2.imread(file_path)]
        convert_coord = is_pdf and not PDFProcessor.is_image_based_pdf(file_path)
        text = ""

        for page, image in enumerate(images):
            img_size = image.size if is_pdf else image.shape[1::-1]
            pdf_size = self.get_pdf_page_size(file_path, page) if convert_coord else img_size
            image_path = f'{os.path.splitext(file_path)[0]}_{page}.jpg' if is_pdf else file_path
            if is_pdf:
                image.save(image_path, 'JPEG')

            text_coordinates, images[page] = self.get_text_coordinates(image_path)
            text = self.get_text_from_image(file_path, page, convert_coord, img_size, pdf_size, text, text_coordinates)

        return images[0], text


class TesseractOCREngine(OCRBase):
    def __init__(self, lang: Union[List[str], str] = 'eng', psm: int = 11):
        super().__init__(lang)
        self.psm = psm

    def perform_ocr(self, file_path: str) -> Tuple[Mat, str]:
        """
        Выполнение OCR с помощью Tesseract.
        :param file_path: Путь к изображению.
        :return: Изображение с выделенными прямоугольниками, распознанный текст.
        """
        import pytesseract
        image = cv2.imread(file_path, 0)
        data = pytesseract.image_to_data(image, lang=self.lang, config=f"--psm {self.psm}", output_type="dict")

        result = []
        current_block = []

        for item in data["text"]:
            if item == '':
                if current_block:
                    result.append(' '.join(current_block))
                    current_block = []
            else:
                current_block.append(item)

        # Добавить последний блок, если он не пустой
        if current_block:
            result.append(' '.join(current_block))

        # Объединить все блоки с переводами строки
        text = '\n'.join(result)

        for i in range(len(data['left'])):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            if data['conf'][i] == -1:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image, text


class PaddleOCREngine(OCRBase):
    def __init__(self, lang: Union[List[str], str] = 'en'):
        super().__init__(lang)
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(lang=lang)

    def perform_ocr(self, file_path: str) -> Tuple[Mat, str]:
        """
        Выполнение OCR с помощью PaddleOCR.
        :param file_path: Путь к изображению.
        :return: Изображение с выделенными прямоугольниками, распознанный текст.
        """
        image = cv2.imread(file_path, 0)
        result = self.ocr.ocr(image)
        all_text = []
        for line in result:
            all_text.extend(text[1][0] for text in line)
        for line in result:
            for box, _ in line:
                points = np.array(box, dtype=np.int32)
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return image, "\n".join(all_text)


class DocTREngine(OCRBase):
    def __init__(self, detect_language: bool = False):
        super().__init__()
        from doctr.models import ocr_predictor
        self.ocr = ocr_predictor(pretrained=True, detect_language=detect_language)

    def perform_ocr(self, file_path: str) -> Tuple[Mat, str]:
        """
        Выполнение OCR с помощью DocTR.
        :param file_path: Путь к изображению.
        :return: Изображение с выделенными прямоугольниками, распознанный текст.
        """
        from doctr.io import DocumentFile
        image = cv2.imread(file_path, 0)
        doc = DocumentFile.from_images(file_path)
        result = self.ocr(doc)
        text = "\n".join(
            [" ".join([word.value for word in line.words])
                for page in result.pages
                for block in page.blocks
                for line in block.lines]
        )
        h, w = image.shape[:2]
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        (x1_norm, y1_norm), (x2_norm, y2_norm) = word.geometry
                        x1, y1, x2, y2 = int(x1_norm * w), int(y1_norm * h), int(x2_norm * w), int(y2_norm * h)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image, text


class BaseProcessor:
    def __init__(
            self,
            file_path: str,
            checkbox: bool,
            selected_engine: str,
            lang_selected: Union[List[str], str] = None,
            only_ocr: bool = False,
            min_confidence: int = 50,
            x_shift: float = 1.0,
            y_shift: float = 0.5,
            psm: int = 11
    ):
        self.file_path = file_path
        self.checkbox = checkbox
        self.lang_selected = lang_selected or ['en']
        self.only_ocr = only_ocr
        self.min_confidence = min_confidence
        self.ocr = self.initialize_ocr(selected_engine, x_shift, y_shift, psm)

    @staticmethod
    def combine_excel_sheets(file_path: str) -> DataFrame:
        """
        Чтение всех листов из Excel-файла и их объединение в один DataFrame.
        :param file_path: Путь к Excel-файлу.
        :return: Содержимое всех листов в виде DataFrame.
        """
        sheets = pd.read_excel(file_path, sheet_name=None)
        dfs = list(sheets.values())
        return pd.concat(dfs, ignore_index=True)

    def initialize_ocr(self, selected_engine: str, x_shift: float, y_shift: float, psm: int):
        """
        Инициализация OCR-движка.
        :param selected_engine: Выбранный OCR-движок.
        :param x_shift: 
        :param y_shift:
        :param psm:
        :return: Экземпляр класса OCR.
        """
        # OCR-движки, где первый элемент - класс для только для OCR, второй - класс для обработки таблиц
        ocr_engines = {
            "EasyOCR": (EasyOCREngine, EasyOCR),
            "TesseractOCR": (TesseractOCREngine, TesseractOCR),
            "PaddleOCR": (PaddleOCREngine, PaddleOCR),
            "DocTR": (DocTREngine, DocTR),
        }

        if selected_engine in ocr_engines:
            ocr_class = ocr_engines[selected_engine][0] if self.only_ocr else ocr_engines[selected_engine][1]
            if selected_engine == "EasyOCR" and ocr_class == EasyOCREngine:
                return ocr_class(lang=self.lang_selected, x_shift=x_shift, y_shift=y_shift)
            elif selected_engine == "TesseractOCR":
                return ocr_class(lang="+".join(self.lang_selected), psm=psm)
            elif selected_engine == "DocTR":
                return ocr_class(detect_language=True)
            else:
                return ocr_class(lang=self.lang_selected)

        return OCRBase(lang=self.lang_selected) if self.only_ocr else None

    def extract_tables(self, obj, path_to_excel: str):
        """
        Извлечение таблиц из PDF файла и сохранение в Excel.
        :param obj: Объект для обработки (PDF или изображение).
        :param path_to_excel: Путь к Excel-файлу.
        :return: Словарь или список с извлеченными таблицами.
        """

        extracted_tables = obj.extract_tables(
            ocr=self.ocr,
            implicit_rows=False,
            borderless_tables=self.checkbox,
            min_confidence=self.min_confidence
        )
        obj.to_xlsx(
            dest=path_to_excel,
            ocr=self.ocr,
            implicit_rows=False,
            borderless_tables=self.checkbox,
            min_confidence=self.min_confidence
        )
        return extracted_tables

    @staticmethod
    def handle_tables(obj, elem, text, dict_boxes):
        """
        Обработка таблиц.
        :param obj: Объект для обработки (PDF или изображение)..
        :param elem: Элемент класса.
        :param text: распознанный текст.
        :param dict_boxes: Словарь, который проверяет, что текст уже не повторялся.
        :return:
        """
        for i in elem.content:
            for cell in elem.content[i]:
                if cell.value and (cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2) not in dict_boxes:
                    text += cell.value.replace("\n", " ") + "\n"
                    dict_boxes[(cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2)] = cell.value
                cv2.rectangle(
                    obj.images[0],
                    [cell.bbox.x1, cell.bbox.y1],
                    [cell.bbox.x2, cell.bbox.y2],
                    (0, 0, 0),
                    2
                )
        return text


class PDFProcessor(BaseProcessor):
    def __init__(
            self,
            file_path: str,
            checkbox: bool,
            selected_engine: str,
            lang_selected: Optional[List[str]] = None,
            only_ocr: bool = False,
            min_confidence: int = 50,
            x_shift: float = 1.0,
            y_shift: float = 0.5,
            psm: int = 11
    ):
        super().__init__(
            file_path, 
            checkbox, 
            selected_engine, 
            lang_selected, 
            only_ocr, 
            min_confidence, 
            x_shift, 
            y_shift,
            psm
        )
        self.pdf = PDF(file_path, detect_rotation=True, pdf_text_extraction=True)
        if not self.is_image_based_pdf(file_path) and not only_ocr:
            self.ocr = None

    @staticmethod
    def is_image_based_pdf(file_path: str) -> bool:
        """
        Проверка, является ли PDF файл изображением.
        :param file_path: Путь к PDF файлу.
        :return: True, если PDF содержит изображения, иначе False.
        """
        doc = pymupdf.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                return False
        return True

    def process(self):
        """
        Основной метод обработки PDF файла.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame, путь к Excel-файлу.
        """
        if self.only_ocr:
            # Выполнить только OCR и вернуть распознанный текст
            image, extracted_text = self.ocr.perform_ocr(self.file_path)
            return image, extracted_text, pd.DataFrame(), ""
        else:
            dict_boxes, text = {}, ""
            path_to_excel = f"{self.file_path}.xlsx"
            for elements in self.extract_tables(self.pdf, path_to_excel).values():
                for elem in elements:
                    text = self.handle_tables(self.pdf, elem, text, dict_boxes)
            cv2.imwrite(f"{self.file_path}_rect.jpg", self.pdf.images[0])
            df = self.combine_excel_sheets(path_to_excel)
            return self.pdf.images[0], text, df, path_to_excel


class ImageProcessor(BaseProcessor):
    def __init__(
            self,
            file_path: str,
            checkbox: bool,
            selected_engine: str,
            lang_selected: Optional[List[str]] = None,
            only_ocr: bool = False,
            min_confidence: int = 50,
            x_shift: float = 1.0,
            y_shift: float = 0.5,
            psm: int = 11
    ):
        super().__init__(
            file_path,
            checkbox,
            selected_engine,
            lang_selected,
            only_ocr,
            min_confidence,
            x_shift,
            y_shift,
            psm
        )
        self.img = Image(file_path, detect_rotation=True)

    def process(self):
        """
        Основной метод обработки изображения.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame, путь к Excel-файлу.
        """
        if self.only_ocr:
            # Выполнить только OCR и вернуть распознанный текст
            image, extracted_text = self.ocr.perform_ocr(self.file_path)
            return image, extracted_text, pd.DataFrame(), ""
        else:
            dict_boxes, text = {}, ""
            path_to_excel = f"{self.file_path}.xlsx"
            for elem in self.extract_tables(self.img, path_to_excel):
                text = self.handle_tables(self.img, elem, text, dict_boxes)
            cv2.imwrite(f"{self.file_path}_rect.jpg", self.img.images[0])
            df = self.combine_excel_sheets(path_to_excel)
            return self.img.images[0], text, df, path_to_excel
