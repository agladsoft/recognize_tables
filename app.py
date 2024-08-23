import os.path

import cv2
import gradio as gr
from gradio_pdf import PDF
from gradio_image_prompter import ImagePrompter
from main import PDFTableProcessor, ImageTableProcessor, ImageBlocksProcessor


def run_tasks_pdf(pdf, checkbox_, lang_selected_):
    image, text_, df, path_to_excel = PDFTableProcessor(pdf, not checkbox_, lang_selected_).process()
    url = f'<a href="file/{path_to_excel}" target="_blank" ' \
          f'rel="noopener noreferrer">{os.path.basename(path_to_excel)}</a>'
    return gr.update(value=image, height=1000), text_, df, url


def run_tasks_image(input_img, checkbox_, lang_selected_):
    image, text_, df, path_to_excel = ImageTableProcessor(
        input_img["background"],
        not checkbox_,
        lang_selected_
    ).process()
    url = f'<a href="file/{path_to_excel}" target="_blank" ' \
          f'rel="noopener noreferrer">{os.path.basename(path_to_excel)}</a>'
    return gr.update(value=image, height=1000), text_, df, url


def run_tasks_image_rect(input_img, checkbox_, lang_selected_):
    image = cv2.imread(input_img["image"])
    for point in input_img["points"]:
        x1, y1, _, x2, y2, _ = point  # Распаковка координат
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(input_img["image"], image)
    image, text_, df, path_to_excel = ImageTableProcessor(input_img["image"], not checkbox_, lang_selected_).process()
    url = f'<a href="file/{path_to_excel}" target="_blank" ' \
          f'rel="noopener noreferrer">{os.path.basename(path_to_excel)}</a>'
    return gr.update(value=image, height=1000), text_, df, url


def run_tasks_pdf_blocks(input_img, x, y, lang_selected_):
    processed_img, text_, df = ImageBlocksProcessor(input_img, x, y, lang_selected_).process()
    return gr.update(value=processed_img, height=1000), text_, df, None


def get_lang():
    return ["en", "ru", "ch_sim", "ch_tra"]


with gr.Blocks() as demo:
    checkbox = gr.Checkbox(
        value=True,
        show_label=True,
        info="Если есть структура у таблицы, ставим галочку",
        label="Прорисованы ли контуры"
    )
    lang_selected = gr.Dropdown(
        choices=get_lang(),
        info="Китайский язык нужно использовать отдельно от других языков или только с английским",
        label="Выберите языки",
        value=["en"],
        multiselect=True
    )

    vertical_offset = 13  # Смещение вниз в пикселях

    with gr.Row():
        with gr.Column():
            with gr.Tab("Загрузка PDF"):
                with gr.Row():
                    with gr.Column():
                        input_pdf = PDF(label="pdf")
                        button_pdf = gr.Button(value="Распознать данные из PDF")

            with gr.Tab("Загрузка JPG (1 editor)"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.ImageEditor(label="jpg", type="filepath")
                        button_image = gr.Button(value="Распознать данные из JPG")

            with gr.Tab("Загрузка JPG (2 editor)"):
                with gr.Row():
                    with gr.Column():
                        input_image_rect = ImagePrompter(label="jpg", type="filepath")
                        button_image_rect = gr.Button(value="Распознать данные из JPG")

            with gr.Tab("Распознавание блоков"):
                x_ths = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Перемещение по X",
                )
                y_ths = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=0.6,
                    step=0.1,
                    interactive=True,
                    label="Перемещение по Y"
                )
                with gr.Row():
                    with gr.Column():
                        input_file = gr.File(label="pdf,jpg", type="filepath")
                        button_file = gr.Button(value="Распознать данные")

        with gr.Column():
            gr.HTML(f'<div style="margin-top:{vertical_offset}px;"></div>')  # Добавление отступа сверху
            output_image = gr.Image(label="Выделение текста или таблиц", type="numpy")

    url_to_excel = gr.HTML()
    table = gr.DataFrame(
        interactive=False,
        wrap=True,
    )
    text = gr.Text()

    button_pdf.click(
        fn=run_tasks_pdf,
        inputs=[input_pdf, checkbox, lang_selected],
        outputs=[output_image, text, table, url_to_excel]
    )
    button_image.click(
        fn=run_tasks_image,
        inputs=[input_image, checkbox, lang_selected],
        outputs=[output_image, text, table, url_to_excel]
    )
    button_image_rect.click(
        fn=run_tasks_image_rect,
        inputs=[input_image_rect, checkbox, lang_selected],
        outputs=[output_image, text, table, url_to_excel]
    )
    button_file.click(
        fn=run_tasks_pdf_blocks,
        inputs=[input_file, x_ths, y_ths, lang_selected],
        outputs=[output_image, text, table, url_to_excel]
    )

    gr.Examples(
        examples=[
            ["/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page0.jpg"],
            ["/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page1.jpg"]
        ],
        fn=run_tasks_image,
        inputs=[input_image, checkbox, lang_selected],
        outputs=[output_image, text, table]
    )

demo.launch()
