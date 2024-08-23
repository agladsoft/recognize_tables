import cv2
import gradio as gr
from gradio_pdf import PDF
from gradio_image_prompter import ImagePrompter
from main import process_img2table_pdf, process_img2table_jpg


def run_tasks_pdf(pdf_, checkbox_, lang_selected_):
    image, text_, df = process_img2table_pdf(pdf_, not checkbox_, lang_selected_)
    return gr.update(value=image, height=1000), text_, df


def run_tasks_image(input_image_, checkbox_, lang_selected_):
    processed_img, text_, df = process_img2table_jpg(input_image_["background"], not checkbox_, lang_selected_)
    return gr.update(value=processed_img, height=1000), text_, df


def run_tasks_image_rect(input_image_, checkbox_, lang_selected_):
    image = cv2.imread(input_image_["image"])
    for point in input_image_["points"]:
        x1, y1, _, x2, y2, _ = point  # Распаковка координат
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(input_image_["image"], image)
    processed_img, text_, df = process_img2table_jpg(input_image_["image"], not checkbox_, lang_selected_)
    return gr.update(value=processed_img, height=1000), text_, df


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

        with gr.Column():
            gr.HTML(f'<div style="margin-top:{vertical_offset}px;"></div>')  # Добавление отступа сверху
            output_image = gr.Image(label="Выделение текста или таблиц", type="numpy")

    table = gr.DataFrame(
        interactive=False,
        wrap=True,
    )
    text = gr.Text()

    button_pdf.click(
        fn=run_tasks_pdf,
        inputs=[input_pdf, checkbox, lang_selected],
        outputs=[output_image, text, table]
    )
    button_image.click(
        fn=run_tasks_image,
        inputs=[input_image, checkbox, lang_selected],
        outputs=[output_image, text, table]
    )
    button_image_rect.click(
        fn=run_tasks_image_rect,
        inputs=[input_image_rect, checkbox, lang_selected],
        outputs=[output_image, text, table]
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
