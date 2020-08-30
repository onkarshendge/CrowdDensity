import cv2
import pafy
import time

DELAY_BETWEEN_IMAGES = 60
BLOB_SCALE_FACTOR = 0.00932
USE_YOUTUBE = True

if __name__ == '__main__':
    currentframe = 0
    if USE_YOUTUBE:
        url = 'https://youtu.be/mRe-514tGMg'
        video = pafy.new(url)
        best = video.streams[3]
        print(best.url)
        camera = cv2.VideoCapture(best.url)

    while True:
        ret, frame = camera.read()
        # name = './samples/timesquare-day/' + str(currentframe) + '.jpg'
        #
        cv2.imshow("Image", frame)
        #
        # cv2.imwrite(name, frame)
        # currentframe += 1

        if cv2.waitKey(40) == 27:
            break
        # time.sleep(DELAY_BETWEEN_IMAGES)

    cv2.destroyAllWindows()
