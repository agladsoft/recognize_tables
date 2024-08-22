import gradio as gr
from gradio_pdf import PDF
from main2 import process_img2table_pdf, process_img2table_jpg


def run_tasks_pdf(pdf_, lang_selected_):
    return process_img2table_pdf(pdf_, lang_selected_)


def run_tasks_image(input_image_, checkbox_, lang_selected_):
    return process_img2table_jpg(input_image_["background"], checkbox_, lang_selected_)


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
        with gr.Column():
            input_pdf = PDF(label="Загрузка pdf")
            button_pdf = gr.Button()
        with gr.Column():
            input_image = gr.ImageEditor(label="Загрузка jpg", type="filepath")
            button_image = gr.Button()
        with gr.Column():
            output_image = gr.Image(label="Выделение текста или таблиц", type="numpy")
    text = gr.Text()

    button_pdf.click(run_tasks_pdf, inputs=[input_pdf, lang_selected], outputs=[output_image, text])
    button_image.click(run_tasks_image, inputs=[input_image, checkbox, lang_selected], outputs=[output_image, text])

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
