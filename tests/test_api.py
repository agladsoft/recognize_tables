import os
import requests

# URL, на который отправляется запрос
url = "http://127.0.0.1:8000/recognize"

# Путь к файлу
file_path = "../table_example.jpg"

# Извлечение названия файла из пути
filename = os.path.basename(file_path)

# Данные (ключи), которые нужно передать вместе с запросом
params = {
    "is_table_bordered": False,
    "selected_engine": "TesseractOCR",
    "selected_languages": ["eng"],
    "is_multiprocess": False,
    "only_ocr": True,
    "confidence": 50,
    "x_shift": 1.0,
    "y_shift": 0.6,
    "psm": 11
}

# Файл, который нужно загрузить
files = {
    "file": (filename, open(file_path, "rb"))
}

# Отправка POST-запроса
response = requests.post(url, params=params, files=files)

# Проверка ответа сервера
if response.status_code == 200:
    print(response.json()["text"], response.json()["tables"])  # Если ответ в формате JSON
else:
    print(f"Ошибка загрузки: {response.status_code}")
    print(response.text)
