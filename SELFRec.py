from data.loader import FileIO


class SELFRec(object):
    def __init__(self, config):
        self.config = config
        self.training_data = FileIO.load_data_set(config['training.set'])
        self.test_data = FileIO.load_data_set(config['test.set'])
        print('Reading data and preprocessing...')

    def execute(self):
        # import the model module
        import_str = 'from model.' + self.config['model.name'] + ' import ' + self.config['model.name']
        exec(import_str)
        recommender = self.config['model.name'] + '(self.config, self.training_data, self.test_data)'
        eval(recommender).execute()
