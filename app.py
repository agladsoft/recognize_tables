import cv2
import gradio as gr
from gradio_pdf import PDF
from gradio_image_prompter import ImagePrompter
from main2 import process_img2table_pdf, process_img2table_jpg


def run_tasks_pdf(pdf_, lang_selected_):
    return process_img2table_pdf(pdf_, lang_selected_)


def run_tasks_image(input_image_, checkbox_, lang_selected_):
    return process_img2table_jpg(input_image_["background"], checkbox_, lang_selected_)


def run_tasks_image_rect(input_image_, checkbox_, lang_selected_):
    image = cv2.imread(input_image_["image"])
    for point in input_image_["points"]:
        x1, y1, _, x2, y2, _ = point  # Распаковка координат
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Рисуем прямоугольник
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(input_image_["image"], image)
    return process_img2table_jpg(input_image_["image"], checkbox_, lang_selected_)


def get_lang():
    return ["en", "ru", "ch_sim", "ch_tra"]


with gr.Blocks() as demo:
    checkbox = gr.Checkbox(value=False, label="borderless_tables")
    lang_selected = gr.Dropdown(
        choices=get_lang(),
        label="Выберите языки",
        value=["en"],
        multiselect=True
    )

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab("Загрузка PDF"):
                with gr.Row():
                    with gr.Column():
                        input_pdf = PDF(label="Загрузка pdf")
                        button_pdf = gr.Button()

            with gr.Tab("Загрузка JPG (1 editor)"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.ImageEditor(label="Загрузка jpg", type="filepath")
                        button_image = gr.Button()

            with gr.Tab("Загрузка JPG (2 editor)"):
                with gr.Row():
                    with gr.Column():
                        input_image_rect = ImagePrompter(show_label=False, type="filepath")
                        button_image_rect = gr.Button()

        with gr.Column(scale=2):
            output_image = gr.Image(label="Выделение текста или таблиц", type="numpy")

    text = gr.Text()

    button_pdf.click(run_tasks_pdf, inputs=[input_pdf, lang_selected], outputs=[output_image, text])
    button_image.click(run_tasks_image, inputs=[input_image, checkbox, lang_selected], outputs=[output_image, text])
    button_image_rect.click(run_tasks_image_rect, inputs=[input_image_rect, checkbox, lang_selected],
                            outputs=[output_image, text])

    gr.Examples(
        examples=[
            ["/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page0.jpg"],
            ["/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page1.jpg"]
        ],
        inputs=[input_image, checkbox, lang_selected],
        outputs=[output_image, text],
        fn=run_tasks_image,
        cache_examples="lazy",
    )

demo.launch()
