import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./img/Lena.png', help='Image path.')
params = parser.parse_args()

img = cv2.imread(params.path)
canvas = img.copy()
original = img.copy()

drawing = False
mode = 'r'  # 'r': rectangle, 'l': line, 'a': arrowed line
start_point = (-1, -1)


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = canvas.copy()
            if mode == 'r':
                cv2.rectangle(temp, start_point, (x, y), (0, 255, 0), 2)
            elif mode == 'l':
                cv2.line(temp, start_point, (x, y), (255, 0, 0), 2)
            elif mode == 'a':
                cv2.arrowedLine(temp, start_point, (x, y), (0, 0, 255), 2)
            cv2.imshow('Lena', temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == 'r':
            cv2.rectangle(canvas, start_point, (x, y), (0, 255, 0), 2)
        elif mode == 'l':
            cv2.line(canvas, start_point, (x, y), (255, 0, 0), 2)
        elif mode == 'a':
            cv2.arrowedLine(canvas, start_point, (x, y), (0, 0, 255), 2)
        cv2.imshow('Lena', canvas)


cv2.namedWindow('Lena')
cv2.setMouseCallback('Lena', mouse_callback)
cv2.imshow('Lena', canvas)

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        mode = 'r'
        print('Mode: Rectangle')
    elif key == ord('l'):
        mode = 'l'
        print('Mode: Line')
    elif key == ord('a'):
        mode = 'a'
        print('Mode: Arrowed Line')
    elif key == ord('w'):
        cv2.imwrite('img/lena_draw.png', canvas)
        print('Saved: lena_draw.png')
    elif key == ord('c'):
        canvas = original.copy()
        cv2.imshow('Lena', canvas)
        print('Cleared')
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
