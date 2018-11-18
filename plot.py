import matplotlib.pyplot as plt

def plot_balance(vsc, rsc):
	print("Plotting balance")
	plt.plot(rsc, vsc)
	fig = plt.figure()
	fig.savefig("balance.png")


def plot_train_conduct(vsc, rsc):
	print("Plotting training conductance")
	plt.plot(rsc, vsc)
	fig = plt.figure()
	fig.savefig("train_conduct.png")


def plot_test_conduct(vsc, rsc):
	print("Plotting test conductance")
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