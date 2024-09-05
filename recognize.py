import os
import re
import cv2
import math
import pymupdf
import numpy as np
import pandas as pd
from cv2 import Mat
from loguru import logger
from scipy import ndimage
from numpy import ndarray
from pypdf import PdfReader
from pandas import DataFrame
from collections import namedtuple
from pdf2image import convert_from_path
from img2table.document import Image, PDF
from typing import Tuple, List, Optional, Union
from img2table.ocr import EasyOCR, TesseractOCR, PaddleOCR, DocTR


class RotateAdjuster:

    @staticmethod
    def rotate(image: ndarray, angle: int, is_right_angle: bool, background: Optional[tuple] = None) -> ndarray:
        """
        Поворот изображения на 90, 180 или 270 градусов.
        :param image: Исходное изображение в виде матрицы.
        :param angle: Угол, на который нужно повернуть.
        :param background: Оттенок серого цвета.
        :param is_right_angle: Прямой ли этот угол
        :return: Матрица перевернутого изображения.
        """
        old_width, old_height = image.shape[:2]
        if is_right_angle:
            angle_radian: float = math.radians(angle)
            width: float = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
            height: float = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
            image_center: Tuple[float, float] = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat: ndarray = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            rot_mat[1, 2] += (width - old_width) / 2
            rot_mat[0, 2] += (height - old_height) / 2
            return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)
        center: Tuple[int, int] = (old_height // 2, old_width // 2)
        img_matrix: ndarray = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, img_matrix,
            (old_height, old_width),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def rotate_image(self, image: str, page_img: int) -> ndarray:
        """
        Поворот изображения на прямой угол (90, 180, 270).
        :param image: Путь к изображению.
        :param page_img: Номер страницы изображения.
        :return:
        """
        import pytesseract
        img: ndarray = cv2.imread(image, 0)
        rotate_img: str = pytesseract.image_to_osd(img, config='--psm 0 -c min_characters_to_try=5')
        angle_rotated_image: int = int(re.search(r'(?<=Orientation in degrees: )\d+', rotate_img)[0])
        logger.info(f"Угол поворота изображения: {angle_rotated_image}. Страница: {page_img}")
        rotated: ndarray = self.rotate(img, angle_rotated_image, is_right_angle=True, background=(0, 0, 0))
        return self.correct_skew(rotated, page_img=page_img, delta=1, limit=60)

    @staticmethod
    def _determine_score(arr: Union[ndarray, cv2.UMat], angle: Union[int, float]) -> float:
        """
        Определяем наилучший результат для угла.
        :param arr: Изображение в виде матрицы.
        :param angle: Угол, на который нужно повернуть.
        :return:
        """
        data: ndarray = ndimage.rotate(arr, angle, reshape=False, order=0)
        histogram: ndarray = np.sum(data, axis=1, dtype=float)
        score: np.array_api.float64 = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score // 100000000

    def _golden_ratio(self, left: int, right: int, delta: float, thresh: Union[ndarray, cv2.UMat]) -> Union[int, float]:
        """
        Определяем лучший угол для поворота (по золотому сечению). Вот ссылка для ознакомления
        https://en.wikipedia.org/wiki/Golden_ratio
        :param left: Минимальный диапазон для нахождения угла.
        :param right: Максимальный диапазон для нахождения угла.
        :param delta: Минимальный диапазон для нахождения угла.
        :param thresh: Изображение в виде матрицы.
        :return: Наилучший найденный угол для поворота.
        """
        res_phi: float = 2 - (1 + math.sqrt(5)) / 2
        x1, x2 = left + res_phi * (right - left), right - res_phi * (right - left)
        f1, f2 = self._determine_score(thresh, x1), self._determine_score(thresh, x2)
        scores: List[float] = []
        angles: List[float] = []
        while abs(right - left) > delta:
            if f1 < f2:
                left, x1, f1 = x1, x2, f2
                x2: float = right - res_phi * (right - left)
                f2: float = self._determine_score(thresh, x2)
                scores.append(f2)
                angles.append(x2)
            else:
                right, x2, f2 = x2, x1, f1
                x1: float = left + res_phi * (right - left)
                f1: float = self._determine_score(thresh, x1)
                scores.append(f1)
                angles.append(x1)
        return (x1 + x2) / 2

    def correct_skew(self, image: ndarray, page_img: int, delta: int, limit: int) -> ndarray:
        """
        Поворот изображения на маленький угол.
        :param image: Путь к изображению.
        :param page_img: Номер страницы изображения.
        :param delta: Шаг для нахождения нужного угла.
        :param limit: Максимальный допустимый угол для поворота.
        :return:
        """
        thresh: Union[ndarray, cv2.UMat] = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        X_Y: namedtuple = namedtuple("X_Y", "x y")
        dict_angle_and_score: dict = {0: X_Y(0, self._determine_score(thresh, 0))}
        best_angle: Union[int, float] = 0
        for angle in range(1, limit, delta):
            dict_angle_and_score[angle] = X_Y(angle, self._determine_score(thresh, angle))
            dict_angle_and_score[-angle] = X_Y(-angle, self._determine_score(thresh, -angle))
            sorted_x_y: list = sorted(dict_angle_and_score.values(), key=lambda xy: xy.y)
            max_value: X_Y = sorted_x_y[-1]
            min_value: X_Y = sorted_x_y[0]
            if max_value.y > min_value.y * 10:
                left: X_Y = dict_angle_and_score.get(max_value.x - 1)
                right: X_Y = dict_angle_and_score.get(max_value.x + 1)
                if left and right:
                    best_angle = self._golden_ratio(left.x, right.x, 0.1, thresh)
                    best_score: float = self._determine_score(thresh, best_angle)
                    if best_score > min_value.y * 100:
                        break
                    else:
                        del dict_angle_and_score[max_value.x]
        logger.info(f"Лучший угол поворота: {best_angle}. Страница: {page_img}")
        return self.rotate(image, best_angle, is_right_angle=False)


class OCRBase:
    def __init__(self, lang: Union[List[str], str]):
        self.lang = lang

    @staticmethod
    def extract_text_from_pdf(file_path: str, page_img: int) -> str:
        """
        Извлечение текста из PDF внутри заданных координат с использованием pdfminer.
        :param file_path: Путь к файлу.
        :param page_img: Номер страницы изображения.
        :return: Извлеченный текст.
        """
        logger.info(
            f"Извлечение текста из PDF файла (представлен в виде текста): {os.path.basename(file_path)}. "
            f"Страница: {page_img}"
        )
        reader = PdfReader(file_path)
        page = reader.pages[page_img]
        return re.sub(
            r'(\s+|\n+)',
            lambda match: match.group()[0]*1,
            page.extract_text(extraction_mode="layout").strip()
        )

    def get_text_from_image(self, image_path: str, page_img: int) -> Tuple[str, ndarray]:
        raise NotImplementedError("Recognize method not implemented!")

    def perform_ocr(self, file_path) -> Tuple[Mat | ndarray, str]:
        """
        Выполнение OCR с помощью движка.
        :param file_path: Путь к изображению.
        :return: Изображение с выделенными прямоугольниками, распознанный текст.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        is_pdf = ext == ".pdf"
        images: List[Image.images] | List[ndarray] = convert_from_path(file_path) if is_pdf else [cv2.imread(file_path)]
        convert_coord = is_pdf and not PDFProcessor.is_image_based_pdf(file_path)
        text = ""

        for page, image in enumerate(images):
            image_path = f'{os.path.splitext(file_path)[0]}_{page}.jpg' if is_pdf else file_path
            if is_pdf:
                image.save(image_path, format='JPEG')
            if convert_coord:
                text += self.extract_text_from_pdf(file_path, page)
            else:
                image_rotated = RotateAdjuster().rotate_image(image_path, page + 1)
                cv2.imwrite(image_path, image_rotated)
                tuple_obj = self.get_text_from_image(image_path, page + 1)
                text += tuple_obj[0]
                images[page] = tuple_obj[1]

        return images[0], text


class EasyOCREngine(OCRBase):
    def __init__(self, lang: Union[List[str], str], x_shift: float = 1.0, y_shift: float = 0.6):
        super().__init__(lang or ['en'])
        import easyocr
        self.ocr = easyocr.Reader(lang)
        self.x_shift = x_shift
        self.y_shift = y_shift

    def get_text_from_image(self, image_path: str, page_img: int) -> Tuple[str, ndarray]:
        """
        Извлечение координат текста из изображения с использованием easyocr.
        :param image_path: Путь к изображению.
        :param page_img: Номер страницы изображения.
        :return: Список координат и распознанного текста, изображение с выделенными прямоугольниками.
        """
        image = cv2.imread(image_path, 0)
        logger.info(f"Извлечение текста из изображения с использованием EasyOCR: {os.path.basename(image_path)}")
        contours = self.ocr.readtext(image, paragraph=True, x_ths=self.x_shift, y_ths=self.y_shift)
        text_coordinates = []
        list_text = []

        for box, text in contours:
            # Получаем координаты
            top_left, top_right, bottom_right, bottom_left = box
            x1, y1 = int(top_left[0]), int(top_left[1])
            x2, y2 = int(bottom_right[0]), int(bottom_right[1])
            text_coordinates.append(((x1, y1, x2, y2), text))
            list_text.append(text)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)

        return '\n'.join(list_text), image


class TesseractOCREngine(OCRBase):
    def __init__(self, lang: Union[List[str], str], psm: int = 11):
        super().__init__(lang or ['eng'])
        self.psm = psm

    def get_text_from_image(self, image_path: str, page_img: int) -> Tuple[str, ndarray]:
        """
        Выполнение OCR с помощью Tesseract.
        :param image_path: Путь к изображению.
        :param page_img: Номер страницы изображения.
        :return: Изображение с выделенными прямоугольниками, распознанный текст.
        """
        import pytesseract
        image = cv2.imread(image_path, 0)
        logger.info(f"Извлечение текста из изображения с использованием Tesseract: {os.path.basename(image_path)}")
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

        return text, image


class PaddleOCREngine(OCRBase):
    def __init__(self, lang: Union[List[str], str]):
        super().__init__(lang or 'en')
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(lang=lang)

    def get_text_from_image(self, image_path: str, page_img: int) -> Tuple[str, ndarray]:
        """
        Выполнение OCR с помощью PaddleOCR.
        :param image_path: Путь к изображению.
        :param page_img: Номер страницы изображения.
        :return: Изображение с выделенными прямоугольниками, распознанный текст.
        """
        image = cv2.imread(image_path, 0)
        logger.info(f"Извлечение текста из изображения с использованием PaddleOCR: {os.path.basename(image_path)}")
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
        return "\n".join(all_text), image


class DocTREngine(OCRBase):
    def __init__(self, detect_language: bool = True):
        super().__init__(lang='')
        from doctr.models import ocr_predictor
        self.ocr = ocr_predictor(pretrained=True, detect_language=detect_language)

    def get_text_from_image(self, image_path: str, page_img: int) -> Tuple[str, ndarray]:
        """
        Выполнение OCR с помощью DocTR.
        :param image_path: Путь к изображению.
        :param page_img: Номер страницы изображения.
        :return: Изображение с выделенными прямоугольниками, распознанный текст.
        """
        from doctr.io import DocumentFile
        image = cv2.imread(image_path, 0)
        logger.info(f"Извлечение текста из изображения с использованием DocTR: {os.path.basename(image_path)}")
        doc = DocumentFile.from_images(image_path)
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
        return text, image


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
            y_shift: float = 0.6,
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
        logger.info("Чтение всех листов из Excel-файла и их объединение в один DataFrame")
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
    def handle_tables(obj, page, elem, text, dict_boxes):
        """
        Обработка таблиц.
        :param obj: Объект для обработки (PDF или изображение).
        :param page: Номер страницы.
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
                    obj.images[page],
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
            y_shift: float = 0.6,
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

    def extract_tables_from_file(self):
        """
        Извлечение таблиц из PDF файла.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame, путь к Excel-файлу.
        """
        logger.info("Выполнение OCR и извлечение таблиц из PDF файла")
        dict_boxes, text = {}, ""
        path_to_excel = f"{self.file_path}.xlsx"
        for page, elements in enumerate(self.extract_tables(self.pdf, path_to_excel).values()):
            for elem in elements:
                text = self.handle_tables(self.pdf, page, elem, text, dict_boxes)
        cv2.imwrite(f"{self.file_path}_rect.jpg", self.pdf.images[0])
        df = self.combine_excel_sheets(path_to_excel)
        return self.pdf.images[0], text, df, path_to_excel

    def process(self):
        """
        Основной метод обработки PDF файла.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame, путь к Excel-файлу.
        """
        if not self.only_ocr:
            return self.extract_tables_from_file()
        # Выполнить только OCR и вернуть распознанный текст
        logger.info("Выполнение только OCR без извлечения таблиц")
        image, extracted_text = self.ocr.perform_ocr(self.file_path)
        return image, extracted_text, pd.DataFrame(), ""


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
            y_shift: float = 0.6,
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

    def extract_tables_from_file(self):
        """
        Извлечение таблиц из изображения.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame, путь к Excel-файлу.
        """
        logger.info("Выполнение OCR и извлечение таблиц из изображения")
        dict_boxes, text = {}, ""
        path_to_excel = f"{self.file_path}.xlsx"
        for page, elem in enumerate(self.extract_tables(self.img, path_to_excel)):
            text = self.handle_tables(self.img, page, elem, text, dict_boxes)
        cv2.imwrite(f"{self.file_path}_rect.jpg", self.img.images[0])
        df = self.combine_excel_sheets(path_to_excel)
        return self.img.images[0], text, df, path_to_excel

    def process(self):
        """
        Основной метод обработки изображения.
        :return: Изображение с выделенными прямоугольниками, текст из таблиц, DataFrame, путь к Excel-файлу.
        """
        if not self.only_ocr:
            return self.extract_tables_from_file()
        # Выполнить только OCR и вернуть распознанный текст
        logger.info("Выполнение только OCR без извлечения таблиц")
        image, extracted_text = self.ocr.perform_ocr(self.file_path)
        return image, extracted_text, pd.DataFrame(), ""
