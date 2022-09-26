#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configargparse
import cv2 as cv
from utils import CvFpsCalc
from gestures import *

import threading
from collections import deque, Counter
import pyttsx3
import time
import nltk

def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])
    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--device", type=int)
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    parser.add('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add("--min_detection_confidence",
               help='min_detection_confidence',
               type=float)
    parser.add("--min_tracking_confidence",
               help='min_tracking_confidence',
               type=float)
    parser.add("--buffer_len",
               help='Length of gesture buffer',
               type=int)

    args = parser.parse_args()

    return args


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode

# use a fixed-length deque to record the gestures and return 1 with highest-freq
def words2Voice(deq,pp):
    last_word = ''
    verb_list = ['VB','VBD','VBN','IN']
    while True:
        curr_word = Counter(list(deq)).most_common(1)[0][0]
        if curr_word != last_word:
            if curr_word == 'it/it\'s':
                if nltk.pos_tag([last_word])[0][1] in verb_list:
                    curr_word = 'it'
                else:
                    curr_word = 'it\'s'
            pp.say(curr_word)
            last_word = curr_word
            pp.runAndWait()
        time.sleep(0.6)



def main():
    # init global vars
    global gesture_buffer
    global gesture_id

    # about 30fps, which means 30 gestures in each sec
    deq = deque(maxlen=15)
    deq.append('')

    # Argument parsing
    args = get_args()
    WRITE_CONTROL = False
    in_flight = False

    # init text-to-voice func
    pp = pyttsx3.init()

    cap = cv.VideoCapture(0)


    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1

    # start the thread of word-to-voice
    t = threading.Thread(target=words2Voice,args=(deq, pp), daemon=True)
    t.start()

    count=0
    while True:
        success, image2 = cap.read()
        fps = cv_fps_calc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(1) & 0xff
        if key == 27:  # ESC
            break
        elif key == ord('k'):
            mode = 0
            WRITE_CONTROL = False
        elif key == ord('n'):
            mode = 1
            WRITE_CONTROL = True
        if WRITE_CONTROL:
            number = -1
            if 48 <= key <= 57:  # 0 ~ 9
                number = key - 48

        # Camera capture
        image = image2
        debug_image, gesture_id = gesture_detector.recognize(image, number, mode)
        gesture_buffer.add_gesture(gesture_id)

        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)

        cv.imshow('Sign Language', debug_image)

        # index -1 will return the last element in the arr
        if gesture_id!=-1:
            curr_words = gesture_detector.get_hand_sign_text(gesture_id)
            deq.append(curr_words)
        else:
            # append empty when no sign detected
            deq.append("")


    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
