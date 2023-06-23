# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import random
import time
import settings
import os
from multiprocessing import Process, cpu_count, Value, Lock


def random_captcha():
    captcha_text = []
    for i in range(settings.MAX_CAPTCHA):
        c = random.choice(settings.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)

def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image

def generate_captchas(path, start, end, process_id, counter, lock):
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(start, end):
        now = str(int(time.time()))
        text, image = gen_captcha_text_and_image()
        filename = text + '_' + now + '.jpg'
        image.save(path + os.path.sep + filename)

        with lock:
            counter.value += 1
            if counter.value % 1000 == 0:
                print(f'Total progress: {counter.value} captchas generated.')



def gen_by_count_and_path(count, path):
    print(f"Will generate {count} png -> {path}")
    num_processes = cpu_count()
    captchas_per_process = count // num_processes

    processes = []
    counter = Value('i', 0)
    lock = Lock()

    for i in range(num_processes):
        start = i * captchas_per_process
        end = (i + 1) * captchas_per_process if i != num_processes - 1 else count
        process = Process(target=generate_captchas, args=(path, start, end, i, counter, lock))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()



def main():
    gen_by_count_and_path(100000, settings.TRAIN_DATASET_PATH)
    gen_by_count_and_path(10000, settings.PREDICT_DATASET_PATH)
    gen_by_count_and_path(5000, settings.TEST_DATASET_PATH)




if __name__ == '__main__':
    main()