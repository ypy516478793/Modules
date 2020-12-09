import datetime, sys, os

def main(save_folder, args):
    bind = lambda x: "--{:s}={:s}".format(str(x[0]), str(x[1]))
    logfile = os.path.join(save_folder, "log")
    # logger = get_logger(logfile, displaying=True, saving=True, debug=False)
    # logger.info("=" * 100)
    # logger.info("Running at: {:s}".format(str(datetime.datetime.now())))
    # logger.info("Working in directory: {:s}\n".format(save_folder))
    # logger.info("Run experiments: ")
    # logger.info("python {:s}".format(" ".join(sys.argv)))
    # logger.info("Full arguments: ")
    # logger.info("{:s}\n".format(" ".join([bind(i) for i in vars(args).items()])))

    sys.stdout = Logger(logfile)
    print_fun()


def print_fun():
    print("=" * 100)
    print("Running at: {:s}".format(str(datetime.datetime.now())))
    print("Working in directory: {:s}\n".format(save_folder))
    print("Run experiments: ")
    print("python {:s}".format(" ".join(sys.argv)))
    print("Full arguments: ")
    # print("{:s}\n".format(" ".join([bind(i) for i in vars(args).items()])))


def get_logger(logpath, displaying=True, saving=True, debug=False):
    import logging
    logger = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        # info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def build_logger(logpath):
    import logging
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', filename=logpath,
                        level=logging.DEBUG, filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler())


if __name__ == '__main__':
    from python.parser import get_args
    save_folder = "./"
    args = get_args()
    main(save_folder, args)


