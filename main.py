from SELFRec import SELFRec
from util.conf import ModelConf


if __name__ == '__main__':
    print('ExpSSGL')
    model = input('Please enter the model you want to run:')
    conf = ModelConf('./conf/' + model + '.conf')
    rec = SELFRec(conf)
    rec.execute()
