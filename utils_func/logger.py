import datetime, sys, os

def main(save_folder, args):
    bind = lambda x: "--{:s}={:s}".format(str(x[0]), str(x[1]))
    logfile = os.path.join(save_folder, "LOG")
    logger = get_logger(logfile, displaying=True, saving=True, debug=False)
    logger.info("=" * 100)
    logger.info("Running at: {:s}".format(str(datetime.datetime.now())))
    logger.info("Working in directory: {:s}\n".format(save_folder))
    logger.info("Run experiments: ")
    logger.info("python {:s}".format(" ".join(sys.argv)))
    logger.info("Full arguments: ")
    logger.info("{:s}\n".format(" ".join([bind(i) for i in vars(args).items()])))

def get_logger(logpath, displaying=True, saving=True, debug=False):
    import logging
    logger = logging.getLogger("new")
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':
    from utils_func.parser import get_args
    save_folder = "./"
    args = get_args()
    main(save_folder, args)


