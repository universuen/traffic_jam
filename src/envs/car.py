class Car:
    def __init__(self, position: int):
        assert position > 0
        self.position = position

    def move(self):
        self.position -= 1

    def __repr__(self):
        return f'Car({self.position})'
