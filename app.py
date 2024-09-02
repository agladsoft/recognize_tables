import os
import cv2
import gradio as gr
from gradio_pdf import PDF
from pandas import DataFrame
from typing import List, Tuple
from main import PDFProcessor, ImageProcessor
from gradio_image_prompter import ImagePrompter
# from gradio_image_annotation import image_annotator


css = """
#btn_theme {
    width: 250px;
    height: 40px;
}
"""

# JavaScript для установки параметра темы и переключения темы
js_func = """
function() {
    const url = new URL(window.location);
    const currentTheme = url.searchParams.get('__theme');

    if (currentTheme === 'dark') {
        url.searchParams.set('__theme', 'light');
    } 
    else if (currentTheme === 'light') {
        url.searchParams.set('__theme', 'dark');
    }
    else {
        url.searchParams.set('__theme', 'light');
    }

    window.location.href = url.href;
}
"""

# Шаблон для ссылки на excel-файл
excel_link_template: str = '<h3>Ссылка на Excel-файл: <a href="file/{0}" target="_blank" ' \
                           'rel="noopener noreferrer">{1}</a></h3>'


def process_pdf(
        pdf_file: str,
        is_table_bordered: bool,
        selected_engine: str,
        selected_languages: List[str],
        only_ocr: bool,
        confidence: int,
        x_shift: float,
        y_shift: float,
        psm: int
) -> Tuple[gr.update, str, DataFrame, str]:
    """
    Обработка PDF файла.
    :param pdf_file: PDF файл.
    :param is_table_bordered: Наличие границ у таблицы.
    :param selected_engine: Выбранный движок.
    :param selected_languages: Выбранные языки.
    :param only_ocr: Нужно распознавать только текст.
    :param confidence: Уверенность распознанного текста.
    :param x_shift: Смещение по x в EasyOCR.
    :param y_shift: Смещение по y в EasyOCR.
    :param psm: Настройка для распознавания текста в Tesseract.
    :return: Обработанное изображение, извлеченный текст, DataFrame, ссылка к Excel-файлу.
    """
    image, extracted_text, dataframe, excel_file_path = PDFProcessor(
        pdf_file,
        not is_table_bordered,
        selected_engine,
        selected_languages,
        only_ocr,
        confidence,
        x_shift,
        y_shift,
        psm
    ).process()
    # Формирование ссылки на excel файл
    excel_link = excel_link_template.format(excel_file_path, os.path.basename(excel_file_path))
    return gr.update(value=image), extracted_text, dataframe, excel_link


def process_image(
        image_data: dict,
        is_table_bordered: bool,
        selected_engine: str,
        selected_languages: List[str],
        only_ocr: bool,
        confidence: int,
        x_shift: float,
        y_shift: float,
        psm: int
) -> Tuple[gr.update, str, DataFrame, str]:
    """
    Обработка изображения.
    :param image_data: Изображение.
    :param is_table_bordered: Наличие границ у таблицы.
    :param selected_engine: Выбранный движок.
    :param selected_languages: Выбранные языки.
    :param only_ocr: Нужно распознавать только текст.
    :param confidence: Уверенность распознанного текста.
    :param x_shift: Смещение по x в EasyOCR.
    :param y_shift: Смещение по y в EasyOCR.
    :param psm: Настройка для распознавания текста в Tesseract.
    :return: Обработанное изображение, извлеченный текст, DataFrame, ссылка к Excel-файлу.
    """
    image, extracted_text, dataframe, excel_file_path = ImageProcessor(
        image_data["background"],
        not is_table_bordered,
        selected_engine,
        selected_languages,
        only_ocr,
        confidence,
        x_shift,
        y_shift,
        psm
    ).process()
    # Формирование ссылки на excel файл
    excel_link = excel_link_template.format(excel_file_path, os.path.basename(excel_file_path))
    return gr.update(value=image), extracted_text, dataframe, excel_link


def process_image_with_rectangles(
        image_data: dict,
        is_table_bordered: bool,
        selected_engine: str,
        selected_languages: List[str],
        only_ocr: bool,
        confidence: int,
        x_shift: float,
        y_shift: float,
        psm: int
) -> Tuple[gr.update, str, DataFrame, str]:
    """
    Чтение изображения и рисование прямоугольников.
    :param image_data: Изображение.
    :param is_table_bordered: Наличие границ у таблицы.
    :param selected_engine: Выбранный движок.
    :param selected_languages: Выбранные языки.
    :param only_ocr: Нужно распознавать только текст.
    :param confidence: Уверенность распознанного текста.
    :param x_shift: Смещение по x в EasyOCR.
    :param y_shift: Смещение по y в EasyOCR.
    :param psm: Настройка для распознавания текста в Tesseract.
    :return: Обработанное изображение, извлеченный текст, DataFrame, ссылка к Excel-файлу.
    """
    image = cv2.imread(image_data["image"])
    for point in image_data["points"]:
        x1, y1, _, x2, y2, _ = point  # Распаковка координат
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(image_data["image"], image)

    # Обработка изображения после добавления прямоугольников
    image, extracted_text, dataframe, excel_file_path = ImageProcessor(
        image_data["image"],
        not is_table_bordered,
        selected_engine,
        selected_languages,
        only_ocr,
        confidence,
        x_shift,
        y_shift,
        psm
    ).process()
    # Формирование ссылки на excel файл
    excel_link = excel_link_template.format(excel_file_path, os.path.basename(excel_file_path))
    return gr.update(value=image), extracted_text, dataframe, excel_link


def get_engines() -> List[str]:
    """
    Получение списка движков.
    :return: Список движков.
    """
    return ["EasyOCR", "TesseractOCR", "PaddleOCR", "DocTR"]


def get_supported_languages(selected_engine: str) -> List[str]:
    """
    Получение списка поддерживаемых языков на основе выбранного движка.
    :param selected_engine: Выбранный движок.
    :return: Список поддерживаемых языков.
    """
    if selected_engine == "EasyOCR":
        return ["en", "ru", "ch_sim", "ch_tra", "ar"]  # Пример языков для EasyOCR
    elif selected_engine == "TesseractOCR":
        return ["eng", "rus", "ch_sim", "ch_tra", "ara"]  # Пример языков для Tesseract
    elif selected_engine == "PaddleOCR":
        return ["en", "chinese_cht", "ch", "ru", "ar"]  # Пример языков для PaddleOCR
    elif selected_engine == "DocTR":
        return ["en"]  # Пример языков для DocTR
    return []


def update_languages(selected_engine: str) -> gr.update:
    """
    Обновляет список поддерживаемых языков в зависимости от выбранного движка.
    :param selected_engine: Выбранный движок.
    :return: Обновленный список языков и настройка для UI.
    """
    supported_languages = get_supported_languages(selected_engine)
    if selected_engine in {"PaddleOCR", "DocTR"}:
        return gr.update(choices=supported_languages, value=supported_languages[0], multiselect=False)
    else:
        return gr.update(
            choices=supported_languages,
            value=supported_languages[0] if supported_languages else [],
            multiselect=True
        )


def update_sliders_interactivity(selected_engine: str, only_ocr: bool):
    """
    Обновляет активность слайдеров x_shift_slider и y_shift_slider.
    :param selected_engine: Выбранный движок.
    :param only_ocr: Флажок простого OCR.
    :return: Обновление активности слайдеров.
    """
    # Определяем активность слайдеров на основе выбранного движка и флага только OCR
    active_settings = {
        ("EasyOCR", True): (True, False, False),
        ("TesseractOCR", True): (False, True, False),
        ("TesseractOCR", False): (False, True, True),
        ("PaddleOCR", True): (False, False, False),
        ("DocTR", True): (False, False, False),
        (None, True): (False, False, False),
    }

    # По умолчанию активируем все слайдеры
    default_settings = (False, False, True)

    # Определяем активность слайдеров
    is_active_easy_ocr, is_active_tesseract, is_active_confidence = active_settings.get((selected_engine, only_ocr),
                                                                                        default_settings)

    return (
        gr.update(interactive=is_active_easy_ocr),
        gr.update(interactive=is_active_easy_ocr),
        gr.update(interactive=is_active_tesseract),
        gr.update(interactive=is_active_confidence)
    )


def validate_languages(selected_engine: str, selected_languages: List[str]) -> gr.update:
    """
    Проверяем, если выбран китайский язык и русский язык одновременно.
    :param selected_engine:
    :param selected_languages:
    :return:
    """
    if selected_engine == "EasyOCR" and ("ch_sim" in selected_languages and "ru" in selected_languages):
        return gr.update(value=get_supported_languages(selected_engine)[0])
    return gr.update(value=selected_languages)


with gr.Blocks(css=css) as demo:
    with gr.Row():
        logo_svg = "<img src='https://i.ibb.co/zJvk1NV/OCR-3.png' width='100px' style='display: inline'>"
        gr.HTML(f"<h1><center>{logo_svg} "
                f"Распознавание таблиц и текста из PDF и изображений</center></h1>")
        gr.DuplicateButton("Переключить тему", elem_id="btn_theme").click(None, js=js_func)
    with gr.Row():
        with gr.Column():
            # Dropdown движка для распознавания
            engine = gr.Dropdown(
                choices=get_engines(),
                info="Разные движки распознают по-разному текст, поэтому выберите наилучший для вас движок",
                label="Выберите движок",
                value=get_engines()[0],
                multiselect=False,
                interactive=True
            )
            # Dropdown для выбора языков
            language_selector = gr.Dropdown(
                choices=get_supported_languages(engine.value),
                info="Больше языков - больше времени на распознавание текста. Выберите только необходимые языки",
                label="Выберите языки",
                value=get_supported_languages(engine.value)[0],
                multiselect=True
            )
        with gr.Column():
            # Checkbox для определения наличия структуры у таблицы
            table_structure_checkbox = gr.Checkbox(
                value=True,
                show_label=True,
                info="Если у таблицы есть структура, установите галочку",
                label="Таблица со структурой"
            )
            # Checkbox для простого распознавания текста
            ocr = gr.Checkbox(
                value=False,
                show_label=True,
                info="Распознать текст",
                label="Простое распознавание текста"
            )
            # Slider уверенности распознанного текста
            min_confidence = gr.Slider(
                minimum=0,
                maximum=100,
                value=50,
                step=1,
                interactive=True,
                label="Уверенность распознанного текста",
            )
        with gr.Column():
            # Slider смещения блоков для EasyOCR
            x_shift_slider = gr.Slider(
                minimum=-10.0,
                maximum=10.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Сдвиг по X для EasyOCR",
            )
            y_shift_slider = gr.Slider(
                minimum=-10.0,
                maximum=10.0,
                value=0.6,
                step=0.1,
                interactive=True,
                label="Сдвиг по Y для EasyOCR"
            )
            # Slider psm для Tesseract
            psm_slider = gr.Slider(
                minimum=1,
                maximum=13,
                value=11,
                step=1,
                interactive=True,
                label="PSM для Tesseract",
            )

    engine.change(
        fn=update_languages,
        inputs=[engine],
        outputs=[language_selector]
    )

    engine.change(
        fn=update_sliders_interactivity,
        inputs=[engine, ocr],
        outputs=[x_shift_slider, y_shift_slider, psm_slider, min_confidence]
    )

    ocr.change(
        fn=update_sliders_interactivity,
        inputs=[engine, ocr],
        outputs=[x_shift_slider, y_shift_slider, psm_slider, min_confidence]
    )

    language_selector.change(
        fn=validate_languages,
        inputs=[engine, language_selector],
        outputs=[language_selector]
    )

    vertical_offset = 13  # Смещение вниз в пикселях

    with gr.Row():
        with gr.Column():
            with gr.Tab("Загрузка PDF"):
                with gr.Row():
                    with gr.Column():
                        pdf_input = PDF(label="Выберите PDF файл")
                        pdf_process_button = gr.Button(value="Распознать данные из PDF")

            with gr.Tab("Загрузка JPG (можно обрезать изображение)"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.ImageEditor(label="Выберите JPG изображение", type="filepath")
                        image_process_button = gr.Button(value="Распознать данные из JPG")

            with gr.Tab("Загрузка JPG (можно рисовать таблицу)"):
                with gr.Row():
                    with gr.Column():
                        image_rect_input = ImagePrompter(label="Выберите JPG изображение", type="filepath")
                        image_rect_process_button = gr.Button(value="Распознать данные из JPG")

            # with gr.Tab("Загрузка JPG (можно рисовать таблицу)"):
            #     with gr.Row():
            #         with gr.Column():
            #             image_rect_input = image_annotator(
            #                     boxes_alpha=0,
            #                     label_list=[""],
            #                     label_colors=[(0, 0, 0)],
            #                     box_thickness=1,
            #                     box_selected_thickness=1,
            #                     disable_edit_boxes=True,
            #                     handle_size=2
            #                 )
            #             image_rect_process_button = gr.Button(value="Распознать данные из JPG")

        with gr.Column():
            gr.HTML(f'<div style="margin-top:{vertical_offset}px;"></div>')  # Добавление отступа сверху
            output_image_display = gr.Image(label="Результат обработки", type="numpy")

    excel_link_display = gr.HTML()
    data_table = gr.DataFrame(
        interactive=False,
        wrap=True,
    )
    extracted_text_display = gr.Textbox(label="Извлеченный текст", show_copy_button=True)

    pdf_process_button.click(
        fn=process_pdf,
        inputs=[
            pdf_input,
            table_structure_checkbox,
            engine, language_selector,
            ocr,
            min_confidence,
            x_shift_slider,
            y_shift_slider,
            psm_slider
        ],
        outputs=[output_image_display, extracted_text_display, data_table, excel_link_display]
    )
    image_process_button.click(
        fn=process_image,
        inputs=[
            image_input,
            table_structure_checkbox,
            engine,
            language_selector,
            ocr,
            min_confidence,
            x_shift_slider,
            y_shift_slider,
            psm_slider
        ],
        outputs=[output_image_display, extracted_text_display, data_table, excel_link_display]
    )
    image_rect_process_button.click(
        fn=process_image_with_rectangles,
        inputs=[
            image_rect_input,
            table_structure_checkbox,
            engine,
            language_selector,
            ocr,
            min_confidence,
            x_shift_slider,
            y_shift_slider,
            psm_slider
        ],
        outputs=[output_image_display, extracted_text_display, data_table, excel_link_display]
    )

    gr.Examples(
        examples=[
            ["text_example.png"],
            ["table_example.jpg"]
        ],
        label="Примеры",
        fn=process_image,
        inputs=[image_input, table_structure_checkbox, language_selector],
        outputs=[output_image_display, extracted_text_display, data_table]
    )

    demo.load(
        fn=update_sliders_interactivity,
        inputs=[engine, ocr],
        outputs=[x_shift_slider, y_shift_slider, psm_slider, min_confidence]
    )


if __name__ == "__main__":
    demo.launch(show_error=True)
