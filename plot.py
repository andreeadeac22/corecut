import matplotlib.pyplot as plt
from constants import *
import seaborn

s = [i[sind] / 100 for i in data] # size of data



def plot_balance(vsc, rsc):
	print("Plotting balance")
	seaborn.set()
	plt.plot(rsc, vsc)
	fig = plt.figure()
	fig.savefig("balance.png")



s = [i[sind]/100 for i in data]
start = min(np.min(x), np.min(y))
    end = max(np.max(x), np.max(y))
    fig = plt.figure()
    ax = fig.add_subplot('111')
    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    sline = np.linspace(start, end, 300)
    ax.plot(sline, sline)
    ax.plot()
    ax.scatter(x, y, s=s)
    fts = 11
    ax.set_xlabel(xlabel, fontsize=fts)
    ax.set_ylabel(ylabel, fontsize=fts)
    ax.set_title(title, fontsize=fts)
    fig.savefig('figs/'+file, bbox_inches='tight')


def plot_train_conduct(vsc, rsc):
	print("Plotting training conductance")
	seaborn.set()
	plt.plot(rsc, vsc)


	fig = plt.figure()
	fig.savefig("train_conduct.png")


def plot_test_conduct(vsc, rsc):
	print("Plotting test conductance")
	seaborn.set()
	plt.plot(rsc, vsc)


	fig = plt.figure()
	fig.savefig("test_conduct.png")


def plot_running_times(vsc, rsc):
	print("Plotting running times")
	plt.plot(rsc, vsc)


	fig = plt.figure()
	fig.savefig("running_times.png")


def main():
	for dataset in dataset_list:
		print()
		#build rsc, vsc
	plot_balance(vsc, rsc)
	plot_train_conduct(vsc, rsc)
	plot_test_conduct(vsc, rsc)
	plot_running_times(vsc, rsc)


if __name__ == "__main__":
	main()