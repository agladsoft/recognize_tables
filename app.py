import os
import cv2
import gradio as gr
import pandas as pd
from gradio_pdf import PDF
from typing import List, Tuple
from gradio_image_prompter import ImagePrompter
from recognize import PDFProcessor, ImageProcessor


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


def get_tables(tables) -> gr.update:
    """
    Получаем таблицы и добавляем их в Tabs.
    :param tables: Таблицы с листами.
    :return: Заполненные данные в таблицах.
    """
    # Если вкладок больше, чем таблиц, скрываем оставшиеся вкладки
    extra_tabs = num_tabs - len(tables)
    tables_list = list(tables.values())
    if extra_tabs < 0:
        tables_list = list(tables.values())[:num_tabs]

    # Создаем обновления: заполняем только для тех вкладок, которые имеют данные
    updates = [gr.update(value=value, interactive=True, wrap=True) for value in tables_list]
    updates += [gr.update(value=pd.DataFrame()) for _ in range(extra_tabs)]
    updates += [gr.update(label=key, visible=True) for key in tables]
    updates += [gr.update(visible=False) for _ in range(extra_tabs)]

    return updates


def get_images(images) -> gr.update:
    """
    Получаем таблицы и добавляем их в Tabs.
    :param images: Таблицы с листами.
    :return: Заполненные данные в таблицах.
    """
    # Если вкладок больше, чем изображений, скрываем оставшиеся вкладки
    extra_tabs = num_tabs - len(images)
    if extra_tabs < 0:
        images = images[:num_tabs]

    # Создаем обновления: заполняем только для тех вкладок, которые имеют данные
    updates = [gr.update(value=value, label="Результат обработки", type="numpy") for value in images]
    updates += [gr.update(value=None) for _ in range(extra_tabs)]
    updates += [gr.update(label=f"Страница {i + 1}", visible=True) for i in range(len(images))]
    updates += [gr.update(visible=False) for _ in range(extra_tabs)]

    return updates


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


def update_sliders_interactivity(
        selected_engine: str,
        only_ocr: bool,
        is_active_table_structure: bool,
        is_multiprocess: bool
) -> Tuple[gr.update, gr.update, gr.update, gr.update, gr.update, gr.update]:
    """
    Обновляет активность слайдеров x_shift_slider и y_shift_slider.
    :param selected_engine: Выбранный движок.
    :param only_ocr: Флажок простого OCR.
    :param is_active_table_structure: Флажок наличия структуры у таблицы.
    :param is_multiprocess: Флажок мультипроцессинга.
    :return: Обновление активности слайдеров.
    """
    # Определяем активность слайдеров на основе выбранного движка и флага только OCR
    active_settings = {
        ("EasyOCR", True): (True, False, False, False, False),
        ("TesseractOCR", True): (False, True, False, False, True),
        ("TesseractOCR", False): (False, True, True, True, False),
        ("PaddleOCR", True): (False, False, False, False, False),
        ("DocTR", True): (False, False, False, False, False),
        (None, True): (False, False, False, False, False),
    }

    # По умолчанию активируем все слайдеры
    default_settings = (False, False, True, True, False)

    # Определяем активность слайдеров
    is_active_easy_ocr, \
        is_active_tesseract, \
        is_active_confidence, \
        is_active_table_structure_inter, \
        is_active_multiprocess = active_settings.get((selected_engine, only_ocr), default_settings)

    return (
        gr.update(interactive=is_active_easy_ocr),
        gr.update(interactive=is_active_easy_ocr),
        gr.update(interactive=is_active_tesseract),
        gr.update(interactive=is_active_confidence),
        gr.update(interactive=is_active_table_structure_inter, value=is_active_table_structure),
        gr.update(interactive=is_active_multiprocess, value=is_multiprocess)
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


# Функции запуска
def process_pdf(
        pdf_file: str,
        is_table_bordered: bool,
        selected_engine: str,
        selected_languages: List[str],
        only_ocr: bool,
        confidence: int,
        x_shift: float,
        y_shift: float,
        psm: int,
        is_multiprocess: bool = False
) -> Tuple[str, dict, str, gr.update, gr.update, gr.update, gr.update]:
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
    :param is_multiprocess: Флаг мультипроцессинга.
    :return: Обработанное изображение, извлеченный текст, DataFrame, ссылка к Excel-файлу.
    """
    images, extracted_text, tables, excel_file_path = PDFProcessor(
        pdf_file,
        not is_table_bordered,
        selected_engine,
        selected_languages,
        only_ocr,
        confidence,
        x_shift,
        y_shift,
        psm,
        is_multiprocess
    ).process()
    # Формирование ссылки на excel файл
    excel_link = excel_link_template.format(excel_file_path, os.path.basename(excel_file_path))
    return (
        extracted_text,
        tables,
        excel_link,
        gr.update(visible=False),
        gr.update(visible=False),
        *get_images(images),
        *get_tables(tables)
    )


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
) -> Tuple[str, dict, str, gr.update, gr.update, gr.update, gr.update]:
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
    images, extracted_text, tables, excel_file_path = ImageProcessor(
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
    return (
        extracted_text,
        tables,
        excel_link,
        gr.update(visible=False),
        gr.update(visible=False),
        *get_images(images),
        *get_tables(tables)
    )


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
) -> Tuple[str, dict, str, gr.update, gr.update, gr.update, gr.update]:
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
    images, extracted_text, tables, excel_file_path = ImageProcessor(
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
    return (
        extracted_text,
        tables,
        excel_link,
        gr.update(visible=False),
        gr.update(visible=False),
        *get_images(images),
        *get_tables(tables)
    )


with gr.Blocks(title="Распознавание данных", css=css) as demo:
    # Количество вкладок, которое может изменяться
    num_tabs = 10  # Например, у нас 10 вкладок, но таблиц только 6
    _tables = gr.State(None)

    # Динамически создаем вкладки
    tabs_df = []
    dataframes = []
    tabs_img = []
    images_ = []

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
            # Checkbox для мультипроцессинга
            multiprocessing_checkbox = gr.Checkbox(
                value=False,
                show_label=True,
                info="Если хотите быстрее распознавать текст, установите галочку",
                label="Использовать все процессоры"
            )
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

        with gr.Column():
            with gr.Tabs() as tab_group:
                for _ in range(num_tabs):
                    with gr.TabItem(visible=False) as tab:
                        image_ = gr.Image(label="Результат обработки", type="numpy")
                        tabs_img.append(tab)
                        images_.append(image_)
                with gr.TabItem(visible=True, label="Страница 1") as tab_test:
                    image_test = gr.Image(label="Результат обработки", interactive=False)

    excel_link_display = gr.HTML()

    with gr.Tabs() as tab_group:
        for _ in range(num_tabs):
            with gr.TabItem(visible=False) as tab:
                df = gr.DataFrame()
                tabs_df.append(tab)
                dataframes.append(df)

    extracted_text_display = gr.Textbox(label="Извлеченный текст", show_copy_button=True, interactive=True)

    engine.change(
        fn=update_languages,
        inputs=[engine],
        outputs=[language_selector]
    )

    engine.change(
        fn=update_sliders_interactivity,
        inputs=[
            engine,
            ocr,
            table_structure_checkbox,
            multiprocessing_checkbox
        ],
        outputs=[
            x_shift_slider,
            y_shift_slider,
            psm_slider,
            min_confidence,
            table_structure_checkbox,
            multiprocessing_checkbox
        ]
    )

    ocr.change(
        fn=update_sliders_interactivity,
        inputs=[
            engine,
            ocr,
            table_structure_checkbox,
            multiprocessing_checkbox
        ],
        outputs=[
            x_shift_slider,
            y_shift_slider,
            psm_slider,
            min_confidence,
            table_structure_checkbox,
            multiprocessing_checkbox
        ]
    )

    language_selector.change(
        fn=validate_languages,
        inputs=[engine, language_selector],
        outputs=[language_selector]
    )

    # noinspection PyTypeChecker
    pdf_process_button.click(
        fn=process_pdf,
        inputs=[
            pdf_input,
            table_structure_checkbox,
            engine,
            language_selector,
            ocr,
            min_confidence,
            x_shift_slider,
            y_shift_slider,
            psm_slider,
            multiprocessing_checkbox
        ],
        outputs=[
            extracted_text_display,
            _tables,
            excel_link_display,
            tab_test,
            image_test
        ] + images_ + tabs_img + dataframes + tabs_df
    )

    # noinspection PyTypeChecker
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
        outputs=[
            extracted_text_display,
            _tables,
            excel_link_display,
            tab_test,
            image_test
        ] + images_ + tabs_img + dataframes + tabs_df
    )

    # noinspection PyTypeChecker
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
        outputs=[
            extracted_text_display,
            _tables,
            excel_link_display,
            tab_test,
            image_test
        ] + images_ + tabs_img + dataframes + tabs_df
    )

    # noinspection PyTypeChecker
    gr.Examples(
        examples=[
            ["text_example.png"],
            ["table_example.jpg"]
        ],
        label="Примеры",
        fn=process_image,
        inputs=[image_input, table_structure_checkbox, language_selector],
        outputs=[
            extracted_text_display,
            _tables,
            excel_link_display,
            tab_test,
            image_test
        ] + images_ + tabs_img + dataframes + tabs_df
    )

    demo.load(
        fn=update_sliders_interactivity,
        inputs=[
            engine,
            ocr,
            table_structure_checkbox,
            multiprocessing_checkbox
        ],
        outputs=[
            x_shift_slider,
            y_shift_slider,
            psm_slider,
            min_confidence,
            table_structure_checkbox,
            multiprocessing_checkbox
        ]
    )


if __name__ == "__main__":
    demo.launch(show_error=True)
