import os
import cv2
import math
import fitz
import easyocr
import pandas as pd
from cv2 import Mat
from numpy import ndarray
from PyPDF2 import PdfReader
from pandas import DataFrame
from img2table.ocr import EasyOCR
from pdf2image import convert_from_path
from img2table.document import Image, PDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal
from typing import Tuple, Any, List, Optional, Dict
from img2table.tables.objects.extraction import ExtractedTable


class ExcelHelper:
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


class PDFTableProcessor:
    def __init__(self, file_path: str, checkbox: bool = False, lang_selected: Optional[list] = None):
        self.file_path = file_path
        self.checkbox = checkbox
        self.lang_selected = lang_selected or ['en']
        self.pdf = PDF(file_path, detect_rotation=True, pdf_text_extraction=True)
        self.ocr = EasyOCR(lang=self.lang_selected) if self.is_image_based_pdf(file_path) else None

    @staticmethod
    def is_image_based_pdf(file_path: str) -> bool:
        """
        Проверка, является ли PDF файл изображением.
        :param file_path: Путь к PDF файлу.
        :return: True, если PDF содержит изображения, иначе False.
        """
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                return False
        return True

    def extract_tables(self, path_to_excel: str) -> Dict[int, List[ExtractedTable]]:
        """
        Извлечение таблиц из PDF файла и сохранение в Excel.
        :param path_to_excel: Путь к Excel-файлу.
        :return: Словарь с извлеченными таблицами.
        """
        extracted_tables = self.pdf.extract_tables(
            ocr=self.ocr,
            implicit_rows=False,
            borderless_tables=self.checkbox,
            min_confidence=50
        )
        self.pdf.to_xlsx(
            dest=path_to_excel,
            ocr=self.ocr,
            implicit_rows=False,
            borderless_tables=self.checkbox,
            min_confidence=50
        )
        return extracted_tables

    def process(self) -> Tuple[Mat | ndarray, str, DataFrame, str]:
        """
        Основной метод обработки PDF файла.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame, путь к Excel-файлу.
        """
        dict_boxes, text = {}, ""
        path_to_excel = f"{self.file_path}.xlsx"
        for elems in self.extract_tables(path_to_excel).values():
            for elem in elems:
                for i in elem.content:
                    for cell in elem.content[i]:
                        if cell.value and (cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2) not in dict_boxes:
                            text += cell.value.replace("\n", " ") + "\n\n"
                            dict_boxes[(cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2)] = cell.value
                        cv2.rectangle(
                            self.pdf.images[0],
                            [cell.bbox.x1, cell.bbox.y1],
                            [cell.bbox.x2, cell.bbox.y2],
                            (0, 0, 0),
                            2
                        )
        cv2.imwrite(f"{self.file_path}_rect.jpg", self.pdf.images[0])
        df = ExcelHelper.combine_excel_sheets(path_to_excel)
        return self.pdf.images[0], text, df, path_to_excel


class ImageTableProcessor:
    def __init__(self, file_path: str, checkbox: bool = False, lang_selected: Optional[list] = None):
        self.file_path = file_path
        self.checkbox = checkbox
        self.lang_selected = lang_selected or ['en']
        self.ocr = EasyOCR(lang=self.lang_selected)
        self.img = Image(file_path, detect_rotation=True)

    def extract_tables(self, path_to_excel: str) -> List[ExtractedTable]:
        """
        Извлечение таблиц из изображения и сохранение в Excel.
        :param path_to_excel: Путь к Excel-файлу.
        :return: Список с извлеченными таблицами.
        """
        extracted_tables = self.img.extract_tables(
            ocr=self.ocr,
            implicit_rows=False,
            borderless_tables=self.checkbox,
            min_confidence=50
        )
        self.img.to_xlsx(
            dest=path_to_excel,
            ocr=self.ocr,
            implicit_rows=False,
            borderless_tables=self.checkbox,
            min_confidence=50
        )
        return extracted_tables

    def process(self) -> Tuple[Mat | ndarray, str, DataFrame, str]:
        """
        Основной метод обработки изображения.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame, путь к Excel-файлу.
        """
        dict_boxes, text = {}, ""
        path_to_excel = f"{self.file_path}.xlsx"
        for elem in self.extract_tables(path_to_excel):
            for i in elem.content:
                for cell in elem.content[i]:
                    if cell.value and (cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2) not in dict_boxes:
                        text += cell.value.replace("\n", " ") + "\n\n"
                        dict_boxes[(cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2)] = cell.value
                    cv2.rectangle(
                        self.img.images[0],
                        [cell.bbox.x1, cell.bbox.y1],
                        [cell.bbox.x2, cell.bbox.y2],
                        (0, 0, 0),
                        2
                    )
        cv2.imwrite(f"{self.file_path}_rect.jpg", self.img.images[0])
        df = ExcelHelper.combine_excel_sheets(path_to_excel)
        return self.img.images[0], text, df, path_to_excel


class ImageBlocksProcessor:
    def __init__(self, file_path: str, x_shift: float, y_shift: float, lang_selected: List[str] = None):
        self.file_path = file_path
        self.lang_selected = lang_selected or ['en']
        self.x_shift = x_shift
        self.y_shift = y_shift

    def convert_pdf_to_image(self) -> Tuple[str, Tuple[int, int]]:
        """
        Конвертация первой страницы PDF в изображение и сохранение его как JPG.
        :return: Расположение изображения и его размеры.
        """
        images = convert_from_path(self.file_path)
        image_path = f'{os.path.splitext(self.file_path)[0]}.jpg'
        images[0].save(image_path, 'JPEG')
        image_size = images[0].size  # (width, height)
        return image_path, image_size

    def get_pdf_page_size(self) -> Tuple[int, int]:
        """
        Получает размеры страницы PDF в точках (points).
        :return: Ширина и высота страницы.
        """
        reader = PdfReader(self.file_path)
        first_page = reader.pages[0]
        width = int(first_page.mediabox.width)
        height = int(first_page.mediabox.height)
        return width, height

    def extract_text_within_coordinates(self, coordinates: Tuple[int, int, int, int]) -> str:
        """
        Извлечение текста из PDF внутри заданных координат с использованием pdfminer.
        :param coordinates: Координаты текста (x1, y1, x2, y2).
        :return: Извлеченный текст.
        """
        x1, y1, x2, y2 = coordinates
        extracted_text = ""

        for page_layout in extract_pages(self.file_path):
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

    def process(self) -> Tuple[Mat | ndarray, str, DataFrame]:
        """
        Основная функция обработки PDF или JPG файла.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame.
        """
        ocr_processor = OCRProcessor(self.file_path, self.x_shift, self.y_shift, lang_selected=self.lang_selected)
        ext = os.path.splitext(self.file_path)[-1].lower()

        # Определяем пути и размеры
        if ext == ".pdf" and not PDFTableProcessor.is_image_based_pdf(self.file_path):
            image_path, img_size = self.convert_pdf_to_image()
            pdf_size = self.get_pdf_page_size()
            convert_coord = True
        elif ext == ".pdf" and PDFTableProcessor.is_image_based_pdf(self.file_path):
            image_path, img_size = self.convert_pdf_to_image()
            pdf_size = img_size
            convert_coord = False
        else:
            image_path = self.file_path
            img_size = cv2.imread(image_path).shape[1::-1]  # (width, height)
            pdf_size = img_size
            convert_coord = False

        # Получаем координаты текста и OCR результат
        text_coordinates, image = ocr_processor.get_text_coordinates(image_path)

        # Извлекаем текст
        text = ""
        for coord, ocr_text in text_coordinates:
            if convert_coord:
                pdf_coord = CoordinateAdjuster.adjust_coordinates(coord, img_size, pdf_size)
                text += self.extract_text_within_coordinates(pdf_coord) + "\n\n"
            else:
                text += ocr_text + "\n\n"

        return image, text, pd.DataFrame()


class OCRProcessor:
    def __init__(self, pdf_path: str, x: float, y: float, lang_selected: List[str] = None):
        self.x = x
        self.y = y
        self.pdf_path = pdf_path
        self.lang_selected = lang_selected or ['en']
        self.reader = easyocr.Reader(self.lang_selected)

    def get_text_coordinates(self, image_path: str) \
            -> Tuple[List[Tuple[Tuple[int, int, int, int], Any]], Mat | ndarray]:
        """
        Извлечение координат текста из изображения с использованием easyocr.
        :param image_path: Путь к изображению.
        :return: Список координат и распознанного текста, изображение с выделенными прямоугольниками.
        """
        image = cv2.imread(image_path, 0)
        contours = self.reader.readtext(image, paragraph=True, x_ths=self.x, y_ths=self.y)
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
