# import data from snap
# directed edges are symmetrized
# and nodes not connected to the largest connected component are removed
# split edges in half---training set and half---test set

import os.path
import pdb
import pickle
import math
import random
import argparse
import scipy as sc
import numpy as np
from tqdm import tqdm

from scipy import sparse
import scipy.sparse.linalg as sp_linalg
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgr
import networkx as nx
import time


dict = {'train_van_conduct': 0,
		'test_van_conduct': 0,
		'train_reg_conduct':0,
		'test_reg_conduct':0,
		'size':0,
		'van_balance':0,
		'reg_balance':0,
		'van_running_time':0,
		'reg_running_time':0}


# call using: sp_linalg.eigs(A_sparse, 3)

def split_data(dataset_name, rebuild=False):

	train_file = dataset_name + "_train.gpickle"
	test_file = dataset_name + "_test.gpickle"

	if os.path.isfile(train_file) and not rebuild:

		G_train = nx.read_gpickle(train_file)
		G_test = nx.read_gpickle(test_file)

	else:
		# Get symmetric
		G = nx.read_edgelist(dataset_name+".txt", nodetype=int)

		# Remove nodes not connected to largest component

		if(nx.number_connected_components(G) ==1):
			print("one component")
			G = max(nx.connected_component_subgraphs(G), key=len)
		else:
			max_component = max(nx.connected_component_subgraphs(G), key=len)
			component_sizes = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
			print("max_component", max_component)
			print("component_sizes", component_sizes)
			print("nx.number_connected_components(G)", nx.number_connected_components(G))

		#pdb.set_trace()
		G = G.to_directed()
		g_edges = list(G.edges())
		print("edges[0]", g_edges[0])
		# Shuffle
		random.shuffle(g_edges)
		# Split in training and testing sets
		no_edges = len(g_edges)
		print(g_edges[0])
		half_len = int(no_edges/2)
		print("half_len", half_len)
		train_content = g_edges[:half_len]
		test_content = g_edges[half_len:no_edges]
		G_train = nx.Graph(train_content)
		G_test = nx.Graph(test_content)

		nx.write_gpickle(G_train, train_file)
		nx.write_gpickle(G_test, test_file)

	return G_train, G_test


def process_dataset(dataset_name):
	G_train, G_test = split_data(dataset_name)

	if(nx.number_connected_components(G_train) > 1):
		print("more than one component")
		G_train = max(nx.connected_component_subgraphs(G_train), key=len)

	print("G_train.number_of_nodes()", G_train.number_of_nodes())
	return G_train, G_test



def compute_vanilla_sc(G, resf, debug=False):
	D = np.array([1/math.sqrt(val) for val in [G.degree()[i] for i in G.nodes()]])
	L = nx.normalized_laplacian_matrix(G)

	start_time = time.time()
	vals, vecs = sp_linalg.eigsh(L, k=8)

	print("vals", vals)
	elapsed = time.time() - start_time
	print("train vanilla time", elapsed, file=resf)
	dict['van_running_time'] = elapsed

	vecs = vecs[:,1] # second eigenvector
	y_vecs = D*vecs

	#yns = [ [y_vecs[i], ]]
	i=0
	yns = []
	for n in G.nodes():
		yns += [[y_vecs[i], n]]
		i +=1
	yns = sorted(yns, key=lambda tup: tup[0])

	total_seq = [el[1] for el in yns]
	random.shuffle(total_seq)
	min_conduct = 1

	for i in tqdm(range(len(y_vecs) - 1), mininterval=10, leave=False, desc='  - (Vanilla)   '):
		seq = total_seq[:(i+1)]
		rest_seq = total_seq[(i+1):]

		if(nx.volume(G, seq) != 0 and nx.volume(G, rest_seq) != 0):
				conduct = nx.algorithms.conductance(G=G, S=seq)
				#print("conduct", conduct)
				if conduct < min_conduct:
					min_cardinal = min(len(seq), len(rest_seq))
					min_conduct = conduct
					min_seq = seq.copy()
		else:
			print("Volume is 0")
			print("seq", seq)

	print("train vanilla min_conduct", min_conduct, file=resf)
	print("train vanilla min_cardinal", min_cardinal, file=resf)

	dict['van_balance'] = min_cardinal

	return min_conduct, min_seq


def get_corecut(G, S, tau, n):
	vol = nx.volume(G,S)
	cut = nx.cut_size(G, S)
	s_size = len(S)
	sc_size = n - s_size
	up = cut + (tau/n)*s_size*sc_size
	down = vol + tau*s_size
	if down == 0:
		print("cut: {cut} ,   up: {up},      vol: {vol},      down: {down} ".format(cut=cut, up=up, vol=vol, down=down))
		return 1
	else:
		return up/down


def compute_regularised_sc(G, resf, debug=False):
	degrees = [val for (node, val) in G.degree()]
	sum_deg = sum(degrees)
	n = len(degrees)
	tau = sum_deg/n

	A = (nx.to_scipy_sparse_matrix(G)).astype(float)

	indices = [i for i in range(n)]
	row = np.array(indices)
	col = row
	data = [ 1/math.sqrt(d + tau) for d in degrees]
	D = csr_matrix((data, (row, col)), shape=(n, n))  #degree matrix

	id_data = np.ones(n)
	I = csr_matrix((id_data, (row, col)), shape=(n, n))  #identity matrix

	L = I - (D@A)@D

	start_time = time.time()
	vals, vecs = sp_linalg.eigsh(L, k=6)
	elapsed =  time.time() - start_time
	print("train regularised time", elapsed, file=resf)

	dict['reg_running_time'] = elapsed

	vecs = vecs[:,1] # second eigenvector
	y_vecs = D*vecs

	i = 0
	yns = []
	for node in G.nodes():
		yns += [[y_vecs[i], node]]
		i += 1
	yns = sorted(yns, key=lambda tup: tup[0])


	total_seq = [el[1] for el in yns]
	min_corecut =  1

	for i in tqdm(range(len(y_vecs) - 1), mininterval=3, leave=False, desc='  - (Regularised)   '):
		seq = total_seq[:(i + 1)]
		rest_seq = total_seq[(i + 1):]

		if (nx.volume(G, seq) < nx.volume(G, rest_seq)):
			corecut = get_corecut(G=G, S=seq, tau=tau, n=n)
			if corecut < min_corecut:
				min_cardinal = min(len(seq), len(rest_seq))
				min_corecut = corecut
				min_seq = seq.copy()
		else:
			corecut = get_corecut(G=G, S=rest_seq, tau=tau, n=n)
			if corecut < min_corecut:
				min_cardinal = min(len(seq), len(rest_seq))
				min_corecut = corecut
				min_seq = rest_seq.copy()

	print("train regularised min_corecut", min_corecut, file=resf)
	print("train regularised min_cardinal", min_cardinal, file=resf)

	dict['reg_balance'] = min_cardinal

	return min_corecut, min_seq



def process_all_datasets(list):
	for i in list:
		process_dataset(list)


def plot():
	print("Plotting")
	# for vsc - rsc
	# number of nodes in the smaller partition set
	# training conductance
	# test conductance
	# running time


def experiments():
	list = [3,5, 6,78,9,4]
	list = list[1:]
	print(list)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-exp', '--experimental', action='store_true')
	parser.add_argument('-file', '--file_name', default="data/p2p-Gnutella08")

	opt = parser.parse_args()

	if opt.experimental:
		experiments()
	else:
		dataset_name = opt.file_name
		G_train, G_test = process_dataset(dataset_name)
		resf = open(dataset_name + "_results.txt", 'w')
		dict_resf = open(dataset_name + "_res.txt", 'wb')


		print("###############   VAN      TRAIN        ###################################")
		print("###############   VAN      TRAIN        ###################################", file=resf)
		train_vsc, min_seq1 = compute_vanilla_sc(G_train, resf=resf)

		print("###############   VAN  TEST        ###################################")
		print("###############   VAN   TEST        ###################################", file=resf)
		test_vsc = nx.algorithms.conductance(G_test, min_seq1)
		print("test vanilla min_corecut", test_vsc, file=resf)

		dict['train_van_conduct'] = train_vsc
		dict['test_van_conduct'] = test_vsc

		print("vanilla train_vsc", train_vsc)
		print("min_seq1", len(min_seq1))
		print("vanilla test_vsc", test_vsc)



		print("###############    REG     TRAIN        ###################################")
		print("###############    REG     TRAIN        ###################################", file=resf)
		train_rsc, min_seq2 = compute_regularised_sc(G_train, resf=resf)

		print("###############  REG  TEST        ###################################")
		print("###############  REG  TEST        ###################################", file=resf)
		test_rsc = nx.algorithms.conductance(G_test, min_seq2)
		print("test regularised min_corecut", test_rsc, file=resf)

		dict['train_reg_conduct'] = train_rsc
		dict['test_reg_conduct'] = test_rsc

		print("regularised train_rsc", train_rsc)
		print("min_seq1", len(min_seq2))
		print("regularised test_rsc", test_rsc)

		dict_resf.write(pickle.dumps(dict))



if __name__ == "__main__":
	main()