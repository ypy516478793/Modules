import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_csv(folder, loss_acc_csv):
    results = pd.read_csv(loss_acc_csv)
    train_acc = results["acc"]
    val_acc = results["val_acc"]
    train_loss = results["loss"]
    val_loss = results["val_loss"]
    
    fig, ax1 = plt.subplots()
    iterations = np.arange(1, len(train_acc) + 1)
    ax1.plot(iterations, train_acc, "b")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("train_acc", color="b")
    ax1.tick_params("y", colors="b")
    
    ax2 = ax1.twinx()
    ax2.plot(iterations, train_loss, "r")
    ax2.set_ylabel("train_loss", color="r")
    ax2.tick_params("y", colors="r")
    # plt.savefig(folder + "train.png")
    
    
    fig2, ax1 = plt.subplots()
    ax1.plot(val_acc, "b")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("valid_acc", color="b")
    ax1.tick_params("y", colors="b")
    
    ax2 = ax1.twinx()
    ax2.plot(val_loss, "r")
    ax2.set_ylabel("valid_loss", color="r")
    ax2.tick_params("y", colors="r")
    # plt.savefig(folder + "validation.png")
    plt.show()
    print("")
    
    
if __name__ == "__main__":
    folder = "./"
    loss_acc_csv = "resnet18_flyCells_noES.csv"
    plot_csv(folder, loss_acc_csv)