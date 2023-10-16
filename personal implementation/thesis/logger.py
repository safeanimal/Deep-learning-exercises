class AverageLogger(object):
    def __init__(self):
        super().__init__()
        self.avg = 0
        self.cnt = 0
        self.sum = 0

    def update(self, x, n=1):
        self.sum += x
        self.cnt += n
        self.avg = self.sum / self.cnt

    def clear(self):
        self.avg = 0
        self.cnt = 0
        self.sum = 0
