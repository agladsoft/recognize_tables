import os
import cv2
import gradio as gr
from gradio_pdf import PDF
from pandas import DataFrame
from typing import List, Tuple
from gradio_image_prompter import ImagePrompter
from main import PDFTableProcessor, ImageTableProcessor, ImageBlocksProcessor

# Шаблон для ссылки на excel-файл
excel_link_template: str = 'Ссылка на Excel-файл: <a href="file/{0}" target="_blank" rel="noopener noreferrer">{1}</a>'


def process_pdf(pdf_file: str, is_table_bordered: bool, selected_languages: List[str]) \
        -> Tuple[gr.update, str, DataFrame, str]:
    """
    Обработка PDF файла.
    :param pdf_file: PDF файл.
    :param is_table_bordered: Наличие границ у таблицы.
    :param selected_languages: Выбранные языки.
    :return:
    """
    image, extracted_text, dataframe, excel_file_path = PDFTableProcessor(pdf_file, not is_table_bordered,
                                                                          selected_languages).process()
    # Формирование ссылки на excel файл
    excel_link = excel_link_template.format(excel_file_path, os.path.basename(excel_file_path))
    return gr.update(value=image, height=1000), extracted_text, dataframe, excel_link


def process_image(image_data: dict, is_table_bordered: bool, selected_languages: List[str]) \
        -> Tuple[gr.update, str, DataFrame, str]:
    """
    Обработка изображения.
    :param image_data: Изображение.
    :param is_table_bordered: Наличие границ у таблицы.
    :param selected_languages: Выбранные языки.
    :return: Обработанное изображение, извлеченный текст, DataFrame, ссылка к Excel-файлу.
    """
    image, extracted_text, dataframe, excel_file_path = ImageTableProcessor(
        image_data["background"],
        not is_table_bordered,
        selected_languages
    ).process()
    # Формирование ссылки на excel файл
    excel_link = excel_link_template.format(excel_file_path, os.path.basename(excel_file_path))
    return gr.update(value=image, height=1000), extracted_text, dataframe, excel_link


def process_image_with_rectangles(image_data: dict, is_table_bordered: bool, selected_languages: List[str]) \
        -> Tuple[gr.update, str, DataFrame, str]:
    """
    Чтение изображения и рисование прямоугольников.
    :param image_data: Изображение.
    :param is_table_bordered: Наличие границ у таблицы.
    :param selected_languages: Выбранные языки.
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
    image, extracted_text, dataframe, excel_file_path = ImageTableProcessor(image_data["image"], not is_table_bordered,
                                                                            selected_languages).process()
    # Формирование ссылки на excel файл
    excel_link = excel_link_template.format(excel_file_path, os.path.basename(excel_file_path))
    return gr.update(value=image, height=1000), extracted_text, dataframe, excel_link


def process_image_blocks(input_file: str, x_shift: float, y_shift: float, selected_languages: List[str]) \
        -> Tuple[gr.update, str, DataFrame, None]:
    """
    Обработка блоков изображения.
    :param input_file: Входной файл.
    :param x_shift: Сдвиг по X.
    :param y_shift: Сдвиг по Y.
    :param selected_languages: Выбранные языки.
    :return: Обработанное изображение, извлеченный текст, DataFrame.
    """
    processed_image, extracted_text, dataframe = ImageBlocksProcessor(input_file, x_shift, y_shift,
                                                                      selected_languages).process()
    return gr.update(value=processed_image, height=1000), extracted_text, dataframe, None


def get_supported_languages() -> List[str]:
    """
    Получение списка поддерживаемых языков.
    :return: Список поддерживаемых языков.
    """
    return ["en", "ru", "ch_sim", "ch_tra"]


def validate_languages(selected_languages: List[str]) -> gr.update:
    """
    Проверяем, если выбран китайский язык и русский язык одновременно
    :param selected_languages:
    :return:
    """
    if ("ch_sim" in selected_languages or "ch_tra" in selected_languages) and "ru" in selected_languages:
        return gr.update(value=["en"], label="Выберите языки (китайский язык нельзя комбинировать с русским)")
    return gr.update(value=selected_languages)


with gr.Blocks() as demo:
    # Checkbox для определения наличия структуры у таблицы
    table_structure_checkbox = gr.Checkbox(
        value=True,
        show_label=True,
        info="Если у таблицы есть структура, установите галочку",
        label="Таблица со структурой"
    )

    # Dropdown для выбора языков
    language_selector = gr.Dropdown(
        choices=get_supported_languages(),
        info="Китайский язык нужно использовать отдельно от других или только с английским",
        label="Выберите языки",
        value=["en"],
        multiselect=True
    )

    language_selector.change(
        fn=validate_languages,
        inputs=language_selector,
        outputs=language_selector
    )

    vertical_offset = 13  # Смещение вниз в пикселях

    with gr.Row():
        with gr.Column():
            with gr.Tab("Загрузка PDF"):
                with gr.Row():
                    with gr.Column():
                        pdf_input = PDF(label="Выберите PDF файл")
                        pdf_process_button = gr.Button(value="Распознать данные из PDF")

            with gr.Tab("Загрузка JPG (1 редактор)"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.ImageEditor(label="Выберите JPG изображение", type="filepath")
                        image_process_button = gr.Button(value="Распознать данные из JPG")

            with gr.Tab("Загрузка JPG (2 редактор)"):
                with gr.Row():
                    with gr.Column():
                        image_rect_input = ImagePrompter(label="Выберите JPG изображение", type="filepath")
                        image_rect_process_button = gr.Button(value="Распознать данные из JPG")

            with gr.Tab("Распознавание блоков"):
                x_shift_slider = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Сдвиг по X",
                )
                y_shift_slider = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=0.6,
                    step=0.1,
                    interactive=True,
                    label="Сдвиг по Y"
                )
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="Выберите PDF или JPG файл", type="filepath")
                        file_process_button = gr.Button(value="Распознать данные")

        with gr.Column():
            gr.HTML(f'<div style="margin-top:{vertical_offset}px;"></div>')  # Добавление отступа сверху
            output_image_display = gr.Image(label="Результат обработки", type="numpy")

    excel_link_display = gr.HTML()
    data_table = gr.DataFrame(
        interactive=False,
        wrap=True,
    )
    extracted_text_display = gr.Text()

    pdf_process_button.click(
        fn=process_pdf,
        inputs=[pdf_input, table_structure_checkbox, language_selector],
        outputs=[output_image_display, extracted_text_display, data_table, excel_link_display]
    )
    image_process_button.click(
        fn=process_image,
        inputs=[image_input, table_structure_checkbox, language_selector],
        outputs=[output_image_display, extracted_text_display, data_table, excel_link_display]
    )
    image_rect_process_button.click(
        fn=process_image_with_rectangles,
        inputs=[image_rect_input, table_structure_checkbox, language_selector],
        outputs=[output_image_display, extracted_text_display, data_table, excel_link_display]
    )
    file_process_button.click(
        fn=process_image_blocks,
        inputs=[file_input, x_shift_slider, y_shift_slider, language_selector],
        outputs=[output_image_display, extracted_text_display, data_table, excel_link_display]
    )

    gr.Examples(
        examples=[
            ["/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page0.jpg"],
            ["/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page1.jpg"]
        ],
        fn=process_image,
        inputs=[image_input, table_structure_checkbox, language_selector],
        outputs=[output_image_display, extracted_text_display, data_table]
    )


if __name__ == "__main__":
    demo.launch()
