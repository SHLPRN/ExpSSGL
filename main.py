from SELFRec import SELFRec
from util.conf import ModelConf
import time


if __name__ == '__main__':
    model = 'ExpSSGL'
    print(model)
    s = time.time()
    conf = ModelConf('./conf/' + model + '.conf')
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
