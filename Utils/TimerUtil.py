from datetime import datetime


class TimerUtil:
    def __init__(self):
        self.time_start_training = datetime.now()
        self.time_start_epoch = datetime.now()
        self.time_end_epoch = datetime.now()
        self.epoch_delta = self.time_end_epoch - self.time_start_epoch
        self.total_delta = self.time_end_epoch - self.time_start_training

    def start_epoch(self):
        self.time_start_epoch = datetime.now()

    def end_epoch(self):
        self.time_end_epoch = datetime.now()

    def print_difference_epoch(self, n_epoch, epochs):
        self.epoch_delta = self.time_end_epoch - self.time_start_epoch
        print("Epoch " + str(n_epoch) + '/' + str(epochs) + ": " + str(self.epoch_delta.total_seconds()))

    def print_difference_from_start(self):
        self.total_delta = self.time_end_epoch - self.time_start_training
        print("Total Time", self.total_delta.total_seconds())

    def get_epoch_delta(self):
        return self.epoch_delta.total_seconds()

    def get_total_delta(self):
        return self.total_delta.total_seconds()

    @staticmethod
    def get_time_string():
        datetime_object = datetime.now()
        time_str = str(datetime_object.year)+ "-" + str(datetime_object.month) + "-" + str(datetime_object.day) + "_" \
                   + str(datetime_object.hour) + "-" + str(datetime_object.minute) + "-" + str(datetime_object.second)
        return time_str
