class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        # training_data/test_data: list of data list, element: [u, i, e](u is user, i is item, both not id)
        self.training_data = training
        self.test_data = test   # can also be validation set if the input is for validation
