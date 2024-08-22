import gradio as gr
from main2 import process_pdf, process_img


def run_tasks(input_image_, checkbox_, lang_selected_):
    # return process_pdf(input_image_["background"], lang_selected_)
    return process_img(input_image_["background"], checkbox_, lang_selected_)


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
        input_image = gr.ImageEditor(label="Загрузка jpg", type="filepath")
        # file = gr.File()
        # input_image = gr.Image(label="Загрузка jpg", type="filepath")
        output_image = gr.Image(label="Выделение текста или таблиц", type="numpy")

    button = gr.Button()
    text = gr.Text()
    button.click(run_tasks, inputs=[input_image, checkbox, lang_selected], outputs=[output_image, text])

    gr.Examples(
        examples=[
            ["/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page0.jpg"],
            ["/home/timur/PycharmWork/project_IDP/directory_files/files/pdf/pdf-page1.jpg"]
        ],
        inputs=[input_image, checkbox, lang_selected],
        outputs=[output_image, text],
        fn=run_tasks,
        cache_examples="lazy",
    )

demo.launch()
