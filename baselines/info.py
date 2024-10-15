import logging
import sys
import os
def get_logger(args):
    logger=logging.getLogger("logger")
    
    print=logging.StreamHandler(sys.stdout)
    logdir=os.path.join(args.pro_path,"log/{}".format(args.baseline))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    countfile=os.path.join(logdir, "count.file")
    if not os.path.isfile(countfile):
        fo = open(countfile, "w")
        fo.write(str(0))
        fo.close()
    
    fo = open(countfile, "r")
    count = int(fo.read())
    fo.close()
    fo = open(countfile, "w")
    fo.write(str(count + 1))
    fo.close()
    args.count=count
    
    logname = os.path.join(logdir, str(count) +"-"+args.data+ ".log")  
    file=logging.FileHandler(logname)
    
    
    formatter = logging.Formatter('%(message)s')

    print.setFormatter(formatter)
    file.setFormatter(formatter)
    logger.addHandler(print)
    logger.addHandler(file)
    logger.setLevel(level=logging.INFO)
    return logger

