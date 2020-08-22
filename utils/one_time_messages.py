
class OneTimeMessages(object):
    def __init__(self):
        self.d = []

    def print_once(self, msg, msg_hash=None):
        msg = str(msg)

        if msg_hash is None:
            msg_hash = msg

        if msg_hash not in self.d:
            print(msg)
            self.d.append(msg_hash)
