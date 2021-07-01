# custom logger
class Logger:
    def __init__(self, active=False):
        self.active = active

    def print(self, message, *argv):
        if self.active:
            print(message.format(*argv))
