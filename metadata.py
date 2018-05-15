class Metadata:
    img = None
    name = None
    embedding = None

    def __init__(self, img, name, embedding):
        self.img = img
        self.name = name
        self.embedding = embedding

    def __get_img(self):
        return self.img