
class OneTimeMessages(object):
    def __init__(self):
        self.d = []

    def print_once(self, msg):
        if msg not in self.d:
            print(msg)
            self.d.append(msg)
