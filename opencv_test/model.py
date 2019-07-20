import cv2

class LenaModel:
    def __init__(self):
        self.fname = './data/lena.jpg'

        """
            https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html
        """
    def execute(self):

        print('cv2 버전: %s' %cv2.__version__)

        original = cv2.imread(self.fname, cv2.IMREAD_COLOR)
        gray = cv2.imread(self.fname, cv2.IMREAD_GRAYSCALE)
        unchanged = cv2.imread(self.fname, cv2.IMREAD_UNCHANGED)

        """
        이미지 읽기에는 3가지 속성값이 있음. 
        대신에 -1, 0, 1을 사용해도 됨.
        """

        cv2.imshow('Original', original)
        cv2.imshow('Gray', gray)
        cv2.imshow('Unchanged', unchanged)
        cv2.waitKey(0)  # 키보드 입력시간 대기 - 0 이면 무한대기

        cv2.destroyAllWindows()


