import matplotlib.pyplot as plt
import pandas
import argparse
import numpy as np
from matplotlib.pyplot import cm

def plot(file_path, **kwargs):
    data = pandas.read_csv(file_path, index_col=None, comment='#')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    count = 0
    count2 = 1
    color = cm.rainbow(np.linspace(0,1,26))

    for key in sorted(data.keys()):
        if count > 0 and not key == 'Evaluation' and not key == 'Training' and not key == 'Task no. :':
            x = np.arange(0,len(data[key]))
            y = data[key]
            c = color[(count - 1)%26]
            plt.subplot(4,3, count2)
            plt.xlabel('Epochs', fontsize=8)
            plt.ylabel(key, fontsize=8)
            plt.plot(x,y, linewidth=2, c=c)
            count2 += 1

        count += 1

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file-path', type=str, default='progress.csv')
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameter
    dict_args = vars(args)
    return dict_args

if __name__ == "__main__":
    args = parse_args()
    plot(**args)
