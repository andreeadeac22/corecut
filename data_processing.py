# import data from snap
# directed edges are symmetrized
# and nodes not connected to the largest connected component are removed
# split edges in half---training set and half---test set

import scipy as sc
import os.path
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sp_linalg
from scipy.sparse import csr_matrix
import scipy.sparse.csgraph as csgr
import networkx as nx
import random
import pdb
import pickle
import argparse


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
	unnorm_L = nx.laplacian_matrix(G)

	A = (nx.to_scipy_sparse_matrix(G)).astype(float)
	D = unnorm_L + A
	L = csgr.laplacian(A, normed=True)

	sqrt_D = csr_matrix.sqrt(D)
	D = sp_linalg.inv(sqrt_D)
	vals, vecs = sp_linalg.eigs(L, k=2)
	y_vecs = D.dot(vecs)
	y_vecs = y_vecs[:,1]  # second eigenvector

	i=0
	yns = []
	for n in G.nodes():
		yns += [[y_vecs[i], n]]
		i +=1

	yns = sorted(yns, key=lambda tup: tup[0])

	min_seq = [yns[0][1]]
	print("minseq", min_seq, file=resf)

	min_conduct = nx.algorithms.conductance(G=G, S=min_seq)
	seq = [yns[0][1]]

	print("min_conduct", min_conduct, file=resf)

	for i in range(1, len(vecs)-1):
		seq += [yns[i][1]]  # check why volume is 0
		if debug:
			print("seq", seq, file=resf)
		conduct = nx.algorithms.conductance(G=G, S=seq)
		if conduct < min_conduct:
			min_conduct = conduct
			min_seq = seq

	print("min_conduct", min_conduct, file=resf)
	print("min_seq", min_seq, file=resf)
	return min_conduct


def get_corecut(G, S, tau, n):
	vol = nx.volume(G,S)
	cut = nx.cut_size(G, S)
	s_size = len(S)
	sc_size = n - s_size
	up = cut + (tau/n)*s_size*sc_size
	down = vol + tau*s_size
	return up/down


def compute_regularised_sc(G, resf, debug=False):

	degrees = [val for (node, val) in G.degree()]
	sum_deg = sum(degrees)
	n = len(degrees)

	tau = sum_deg/n


	unnorm_L = nx.laplacian_matrix(G)
	A = (nx.to_scipy_sparse_matrix(G)).astype(float)

	indices = [i for i in range(n)]
	row = np.array(indices)
	col = row
	data = [ d + tau/n for d in degrees]
	D = csr_matrix((data, (row, col)), shape=(n, n))  #degree matrix

	sqrt_D = csr_matrix.sqrt(D)
	D = sp_linalg.inv(sqrt_D) # D ^ (-1/2)

	id_data = np.ones(n)
	I = csr_matrix((id_data, (row, col)), shape=(n, n))  #identity matrix

	#L = I - csr_matrix(csr_matrix(D).multiply(csr_matrix(A))).multiply(csr_matrix(D)) #normed tau laplacian
	L = I - (D@A)@D
	#print("L", L)

	vals, vecs = sp_linalg.eigs(L, k=2)
	y_vecs = D.dot(vecs)
	y_vecs = y_vecs[:, 1]  # second eigenvector

	i = 0
	yns = []
	for node in G.nodes():
		yns += [[y_vecs[i], node]]
		i += 1

	yns = sorted(yns, key=lambda tup: tup[0])

	min_seq = [yns[0][1]]
	print("minseq", min_seq, file=resf)

	min_corecut = get_corecut(G=G, S=min_seq, tau=tau, n=n)
	seq = [yns[0][1]]

	print("min_corecut", min_corecut, file=resf)

	for i in range(1, len(vecs) - 1):
		seq += [yns[i][1]]
		if debug:
			print("seq", seq, file=resf)
		corecut = get_corecut(G=G, S=seq, tau=tau, n=n)
		if corecut < min_corecut:
			min_corecut = corecut
			min_seq = seq

	print("min_corecut", min_corecut, file=resf)
	print("min_seq", min_seq, file=resf)
	return min_corecut



def process_all_datasets(list):
	for i in list:
		process_dataset(list)


def plot():
	# number of nodes in the smaller partition set

def experiments():
	dataset_name = "data/ca-GrQc"
	G_train, G_test = process_dataset(dataset_name)
	resf = open(dataset_name + "_results.txt", 'w')
	compute_regularised_sc(G_test, resf, debug=True)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-exp', '--experimental', action='store_true')

	opt = parser.parse_args()

	if opt.experimental:
		experiments()
	else:
		dataset_name = "data/ca-GrQc"
		G_train, G_test = process_dataset(dataset_name)
		resf = open(dataset_name + "_results.txt", 'w')

		print("###############    REG     TRAIN        ###################################")
		print("###############    REG     TRAIN        ###################################", file=resf)
		train_rsc = compute_regularised_sc(G_train, resf=resf)

		print("###############  REG  TEST        ###################################")
		print("###############  REG  TEST        ###################################", file=resf)
		test_rsc = compute_regularised_sc(G_test, resf=resf)
		print("regularised train_rsc", train_rsc)
		print("regularised test_rsc", test_rsc)




		print("###############   VAN      TRAIN        ###################################")
		print("###############   VAN      TRAIN        ###################################", file=resf)
		train_vsc = compute_vanilla_sc(G_train, resf=resf)

		print("###############   VAN  TEST        ###################################")
		print("###############   VAN   TEST        ###################################", file=resf)
		test_vsc = compute_vanilla_sc(G_test, resf=resf, debug=False)
		print("vanilla train_vsc", train_vsc)
		print("vanilla test_vsc", test_vsc)




if __name__ == "__main__":
	main()