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

dataset_list = [
	"amazon0302",
	"amazon0312",
	"amazon0505",
	"amazon0601",
	"ca-AstroPh",
	"ca-CondMat",
	"ca-GrQc",
	"ca-HepPh",
	"ca-HepTh",
	"cit-HepPh",
	"cit-HepTh",
	"com-amazon.ungraph",
	"com-youtube.ungraph",
	"email-EuAll",
	"email-Eu-core",
	"facebookcombined",
	"p2p-Gnutella04",
	"p2p-Gnutella05",
	"p2p-Gnutella06",
	"p2p-Gnutella08",
	"p2p-Gnutella09",
	"p2p-Gnutella24",
	"p2p-Gnutella25",
	"p2p-Gnutella30",
	"p2p-Gnutella31",
	"roadNet-CA",
	"roadNet-PA",
	"roadNet-TX",
	"soc-Epinions1",
	"soc-Slashdot0811",
	"soc-Slashdot0902",
	"twitter-combined",
	"web-Google",
	"web-NotreDame",
	"web-Stanford",
	"wiki-Talk",
	"wiki-Vote"
]

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