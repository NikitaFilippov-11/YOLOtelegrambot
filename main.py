# import easyocr
import telebot
from telebot import types
import torch
import os
import shutil
import cv2
import moviepy.editor as moviepy
from ultralytics import YOLO
# from minio import Minio
# from minio.error import S3Error
# import io
# from PIL import Image

bot = telebot.TeleBot('6836609256:AAGPOq5sZrj35-vpilWjtS7-QvUTRehOpBw')
# client = Minio(
#     "play.min.io",
#     access_key="L1PTTCA0JGQ2WIXFD9PC",
#     secret_key="3NTQiujIyYmBT9BdE2iUFklMNhP0O3KzTJqVWcjc",
# )
# model = YOLO('yolov8n.pt')
# model = YOLO('Cars.pt')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = None
test_word = None
# URL = 'C:/Users/Nikita/Desktop/'
URL = '/usr/share/app/'
# reader = easyocr.Reader(['en', 'ru'])

# found = client.bucket_exists('yolo-telegram-bot')
# if not found:
#     client.make_bucket('yolo-telegram-bot')
# else:
#     print("Bucket 'yolo_telegram_bot' already exists")


@bot.message_handler(commands=['start'])
def start(message):
    # markup = types.ReplyKeyboardMarkup()
    # button_1 = types.KeyboardButton('YOLOv8 all object')
    # button_2 = types.KeyboardButton('YOLOv5 all object')
    # button_3 = types.KeyboardButton('YOLOv8 license plate')
    # markup.add(button_1, button_2, button_3)
    markup = main_menu_button(message)
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}', reply_markup=markup)
    # bot.register_next_step_handler(message, on_click)


def main_menu_button(message):
    markup = types.ReplyKeyboardMarkup()
    button_1 = types.KeyboardButton('YOLOv8 all object')
    button_2 = types.KeyboardButton('YOLOv5 all object')
    button_3 = types.KeyboardButton('YOLOv8 license plate')
    markup.add(button_1, button_2, button_3)
    bot.send_message(message.chat.id, 'Выберете модель:', reply_markup=markup)
    return markup


def back_to_the_main_menu(message):
    markup = types.ReplyKeyboardMarkup()
    button_return = types.KeyboardButton('Вернуться к выбору модели')
    markup.add(button_return)
    to_pin = bot.send_message(message.chat.id, f'Установленна модель: {message.text}', reply_markup=markup).message_id
    bot.pin_chat_message(message.chat.id, to_pin)


@bot.message_handler(content_types=['text'])
def model_selection(message):
    global model
    global test_word
    if message.text == 'YOLOv8 all object':
        model = YOLO('yolov8n.pt')
        test_word = message.text
        back_to_the_main_menu(message)
    elif message.text == 'YOLOv5 all object':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        test_word = message.text
        back_to_the_main_menu(message)
    elif message.text == 'YOLOv8 license plate':
        model = YOLO('Cars.pt')
        test_word = message.text
        back_to_the_main_menu(message)
    elif message.text == 'Вернуться к выбору модели':
        model = None
        bot.unpin_all_chat_messages(message.chat.id)
        main_menu_button(message)
    else:
        bot.send_message(message.chat.id, f'Такой команды не существует :(')
        # main_menu_button(message)


# @bot.message_handler(content_types=['text'])
# def back_to_the_main_menu(message):
#     if message.text == 'Вернуться к выбору модели':
#         start()


@bot.message_handler(content_types=['photo'])
def get_photo(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = URL + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        # client.put_object('yolo-telegram-bot', file_info.file_path, io.BytesIO(downloaded_file),
        #                   length=file_info.file_size,
        #                   content_type="image/jpeg")
        # # client.fget_object('yolo-telegram-bot', file_info.file_path, src)
        # try:
        #     response = client.get_object('yolo-telegram-bot', file_info.file_path)
        # finally:
        #     response.close()
        #     response.release_conn()
        # src = Image.open(io.BytesIO(downloaded_file))
        send_photo(message, file_info, src)
    except Exception as e:
        bot.reply_to(message, e)


def send_photo(message, file_info, src):
    number = file_info.file_path[file_info.file_path.find('_') + 1:]
    number = '.'.join(number.split('.')[:-1])
    model_file = ai_model_for_photo_processing(src, number)
    if test_word == 'YOLOv8 license plate':
        results = reading_text_from_an_image(number)
    elif test_word == 'YOLOv8 all object':
        results = f'{model_file[0].boxes.cls}'
    else:
        results = model_file.pandas().xyxy[0]
    send_src = URL + f'photos/file_{number}/file_{number}.jpg'
    bot.reply_to(message, f'Фотография обработана!')
    bot.send_photo(message.chat.id, open(send_src, 'rb'), caption=f'{results}')
    remove_photo(number, src)


def ai_model_for_photo_processing(src, number):
    if test_word == 'YOLOv8 license plate' or test_word == 'YOLOv8 all object':
        model_file = model.predict(source=src, save=True, conf=0.5, name=URL + f'photos/file_{number}', save_crop=True)
    else:
        model_file = model(src)
        # b = bytearray(model_file.file[0])
        # client.put_object('yolo-telegram-bot', f'photos/file_{number}', io.BytesIO(downloaded_file),
        #                   length=file_info.file_size,
        #                   content_type="image/jpeg")
        model_file.save(labels=True, save_dir=URL + f'photos/file_{number}')
    return model_file


def remove_photo(number, src):
    os.remove(src)
    shutil.rmtree(URL + f'photos/file_{number}')


@bot.message_handler(content_types=['video'])
def get_video(message):
    try:
        file_info = bot.get_file(message.video.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = URL + file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        send_video(message, file_info, src)
    except Exception as e:
        bot.reply_to(message, e)


def send_video(message, file_info, src):
    number = file_info.file_path[file_info.file_path.find('_') + 1:]
    number = '.'.join(number.split('.')[:-1])
    ai_model_for_video_processing(src, number)
    send_src = URL + f'videos/model_videos/file_{number}.mp4'
    bot.reply_to(message, f'Видео обработано!')
    clip = moviepy.VideoFileClip(send_src)
    compressed = URL + f'videos/send/file_{number}.mp4'
    clip.write_videofile(compressed)
    bot.send_video(message.chat.id, open(compressed, 'rb'))
    remove_video(send_src, src, compressed, number)


def ai_model_for_video_processing(src, number):
    cap = cv2.VideoCapture(src)
    assert cap.isOpened()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_cod = cv2.VideoWriter_fourcc(*'XVID')
    video_output = cv2.VideoWriter(f'/usr/share/app/videos/model_videos/file_{number}.mp4', video_cod, fps,
                                   (frame_width,
                                    frame_height))
    shot = 1
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while shot < frame_number:
        ret, frame = cap.read()
        assert ret

        if test_word == 'YOLOv8 license plate' or test_word == 'YOLOv8 all object':
            results = model(frame)
            annotate_frame = results[0].plot()
        else:
            frame = cv2.resize(frame, (frame_width, frame_height))
            results = score_frame(frame)
            annotate_frame = plot_boxes(results, frame)

        video_output.write(annotate_frame)
        shot += 1

    video_output.release()
    cv2.destroyAllWindows()


def score_frame(frame):
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord


def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.3:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, model.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


def remove_video(send_src, src, compressed, number):
    os.remove(src)
    os.remove(send_src)
    os.remove(compressed)
    if os.path.exists(URL + f'photos/file_{number}'):
        shutil.rmtree(URL + f'photos/file_{number}')


def reading_text_from_an_image(number):
    img = cv2.imread(f'C:/Users/Nikita/Desktop/photos/file_{number}/crops/license-plate/file_{number}.jpg')
    # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # results = reader.readtext(img, allowlist='0123456789ABEKMHOPCTLR')
    results = 1
    return results


bot.polling(none_stop=True)
