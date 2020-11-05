import argparse

def get_args():

    from ast import literal_eval
    list_parser = lambda x: [int(i) for i in x.strip("[]").split(",")] if x else []

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', type=str, help='SEAMII, BP2004, SEAMII_arid, SEAMII_fh', default="SEAMII")
    parser.add_argument('--data_dir', type=str, help='data file directory', default="SEAMII/data_1000_to_1200.h5")
    parser.add_argument('--save_dir', type=str, help="directory of saved results", default="results/")
    parser.add_argument('--epochs', type=int, help='number of epochs', default=30)
    parser.add_argument('--batchsize', type=int, help='batch size', default=2)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    parser.add_argument('--train', type=eval, help='train or test', default=True)
    parser.add_argument('--gpu', type=str, help='which gpu to use', default="1")
    parser.add_argument('--load', type=eval, help='whether load model', default=True)
    parser.add_argument('--load_other', type=str, help='load trained model from other dataset', default="")
    parser.add_argument('--log', type=eval, help='whether to use log to scale freq domain', default=False)
    parser.add_argument('--complex', type=eval, help='whether to use complex freq domain', default=False)
    parser.add_argument('--domain', type=str, help='freq or time', default="time")
    parser.add_argument('--strategy', type=str, help='skip1, random42, random, mix', default="random42")
    parser.add_argument('--missRatio', type=float, help='missing ratio', default=0.875)
    parser.add_argument('--norm', type=str, help='divideMax or scale', default="scale")
    parser.add_argument('--scaleRatio', type=eval, help='scale ratio', default=10e9)
    parser.add_argument('--extraStr', type=str, help='extraStr for saving', default="")
    parser.add_argument('--savefig', type=eval, help='whether to save fig', default=False)

    parser.add_argument('--zoom', type=list_parser, help='zoom section (lh, rh, lw, rw)', default="[0, 150, 960, 1280]")
    parser.add_argument('--hidden_units', type=literal_eval, default='(40, 40)',help='(40, 40), (40, 40, 40) or (40,)')

    parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
    parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
    parser.set_defaults(uncertainty_flag=False)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu