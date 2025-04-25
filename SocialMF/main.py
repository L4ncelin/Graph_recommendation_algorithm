from rec_exec import rec_exec
from util.config import ModelConf

if __name__ == '__main__':
    import time
    s = time.time()
    try:
        conf = ModelConf('./config/' + "SocialMF" + '.conf')
    except KeyError:
        print('wrong num!')
        exit(-1)
    recSys = rec_exec(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
