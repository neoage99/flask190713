from simple_classification_algorithm.model import IrisModel

class IrisController:
    def __init__(self):
        self.model = IrisModel()

    @staticmethod
    def print_menu():
        print('0. 종료')
        print('1. 아이리스 데이터 출력')

    @staticmethod
    def show(param):
        print('RESULT : %s ' % param)

    def run(self):
        model = self.model
        while 1:
            menu = self.print_menu()
            if menu == 0:
                break
            elif menu == 1:
                self.show(model.get_iris())