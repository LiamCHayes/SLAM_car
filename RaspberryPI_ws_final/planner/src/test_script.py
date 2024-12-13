#!/usr/bin/env python3

from gpiozero import Motor
import time

left_side_front = Motor(forward=27,backward=14)
left_side_back = Motor(forward=17,backward=18)
right_side_front = Motor(forward=19,backward=16)
right_side_back = Motor(forward=26, backward=20)

time.sleep(5)
print("Let's goooo!")
count=1
left_side_front.forward()
left_side_back.forward()
right_side_front.forward()
right_side_back.forward()

time.sleep(3)
print('sleeping')
left_side_front.stop()
left_side_back.stop()
right_side_front.stop()
right_side_back.stop()

time.sleep(5)
print('spinninggggg')
left_side_front.forward()
left_side_back.forward()
right_side_front.backward()
right_side_back.backward()

time.sleep(10)
print('stopping')
left_side_front.stop()
left_side_back.stop()
right_side_front.stop()
right_side_back.stop()


