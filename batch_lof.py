#
import sys, os, argparse, random


def	read_tests(FF):
	f = open(FF, "r")
	lines = f.readlines()
	f.close()

	# Usual (base) file
	Bs = []
	# Number of vectors to select from the usual file
	nBs = []
	# List of anomaly files
	As = []
	# List of vectors to chose from each of the anomaly files
	nAs = []
	for i in range(len(lines)):
		if lines[i][0] != '#' and len(lines[i]) > 2:
			#print("lines = ", i, lines[i])
			xx = lines[i].split('\t')
			Bs.append(xx[0])
			nBs.append(int(xx[1]))
			T = []
			S = []
			for j in range(2, len(xx), 2):
				T.append(xx[j])
				S.append(int(xx[j+1]))
			As.append(T)
			nAs.append(S)

	return [Bs, nBs, As, nAs]

def	read_data(FF, num):
	f = open(FF, "r")
	x = f.readlines()
	f.close()

	D = []
	for i in x:
		xx = i.split('\t')
		tmp = []
		for jj in xx:
			tmp.append(float(jj))
		D.append(tmp)

	Dx = random.choices(D, k=num)

	return Dx

def	save_data(FF, D):
	ln = len(D[0])
	f = open(FF, "w")
	for V in D:
		for i in range(ln-1):
			f.write(str(V[i]) + "\t")
		f.write( str(V[ln-1]) )
		f.write("\n")
	f.close()


def	read_info(FF):
	f = open(FF, "r")
	x = f.readlines()
	f.close()

	I = []
	for ln in x:
		xx = ln.split('\t')
		tmp = []
		for i in range(4):
			tmp.append(str(xx[i]))
		tmp.append( str(xx[4][0:len(xx[4])-1]) )
		I.append(tmp)

	return I

def	save_info(FF, u, a, pi, r, I):
	ln = len(I)
	f = open(FF, "a")
	for i in range(ln):
		f.write(u + "_anom_" + a + "\t" + str(pi) + "\t" + str(r) + "\t")
		for j in range(4):
			f.write(str(I[i][j]) + "\t")
		f.write( str(I[i][4]) + "\n" )
	f.close()


def	read_stats(FF):
	f = open(FF, "r")
	x = f.readlines()
	f.close()

	Stats = {}
	ln = len(x)
	for i in range(1,ln):
		xx = x[i].split('\t')
		recall = float(xx[1])
		specificity = float(xx[2])
		accuracy = float(xx[3])
		Stats[xx[0]] = [recall, specificity, accuracy]

	return Stats

def	save_stats(FF, i, Bs, nBs, As, nAs, k, totAn, Dists, Stats):
	f = open(FF, "a")

	f.write(str(i) + "\t" + Bs + "\t" + str(nBs) + "\t")
	ln = len(As)
	for i in range(ln-1):
		f.write(As[i] + ":" + str(nAs[i]) + "_")
	f.write(As[ln-1] + ":" + str(nAs[ln-1]) + "\t")
	f.write(str(totAn) + "\t")
	f.write(str(k) + "\t")
	for ds in Dists:
		f.write(ds + "\t" + str(Stats[ds][0]) + "\t" + str(Stats[ds][1]) + "\t" + str(Stats[ds][2]) + "\t")

	f.write("\n")

	f.close()



"""
python3 batch_lof.py  -d tests_base_gauss_n_20_1.csv  -o info_statistics_gauss_n_20_1.csv

python3 /home/antonioneme/Dropbox/anomDet_CDA/progs/lof_cda_several_class.py  -i  /home/antonioneme/Dropbox/anomDet_CDA/data/PDF/joint_ex_gauss_p_0_9_q_0_2_anom_ex_gauss_p_0_8_q_0_2_anom_gauss_p_0_8_q_0_3_anom_gauss_p_0_9_q_0_3_n_20_x_1.csv  -k 6  -u 50   -o /home/antonioneme/Dropbox/anomDet_CDA/data/PDF/lof_k_6_joint_ex_gauss_p_0_9_q_0_2_anom_ex_gauss_p_0_8_q_0_2_anom_gauss_p_0_8_q_0_3_anom_gauss_p_0_9_q_0_3_n_20_x_1.csv  -o2 /home/antonioneme/Dropbox/anomDet_CDA/data/PDF/stats_lof_k_6_joint_ex_gauss_p_0_9_q_0_2_anom_ex_gauss_p_0_8_q_0_2_anom_gauss_p_0_8_q_0_3_anom_gauss_p_0_9_q_0_3_n_20_x_1.csv  -o3 /home/antonioneme/Dropbox/anomDet_CDA/data/PDF/class_lof_k_6_joint_ex_gauss_p_0_9_q_0_2_anom_ex_gauss_p_0_8_q_0_2_anom_gauss_p_0_8_q_0_3_anom_gauss_p_0_9_q_0_3_n_20_x_1.csv

"""

parser = argparse.ArgumentParser()

parser.add_argument('-d', action = "store", dest = "d", help = "The input file containing the list of usual and anomaly files")
parser.add_argument('-o', action = "store", dest = "o", help = "The output file containing the classification statistics")

args = parser.parse_args()

#Bs: list of usual (base) files
#nBs: list of vectors to extract from each of the base files
#As: list of lists of the anomaly files to comapre to the base file 
#nAs: list of lists of the number of vectors to be extracted from the anomaly files
[Bs, nBs, As, nAs] = read_tests(args.d)

nK = [2, 3, 4, 5]
#nK = [2, 3, 4, 5, 6, 7]
##nK = [2, 3]

print ("Bs = ", Bs, nBs, As, nAs)

#Dists = ['euclidean', 'manhattan', 'wasserstein', 'cosine', 'aitchison', 'hellinger', 'fhr', 'js']
Dists = ['euclidean', 'manhattan', 'wasserstein', 'cosine', 'aitchison', 'hellinger', 'fhr', 'js', 'hilbert']

numT = len(Bs)
for i in range(numT):
	print("Bs = ", i, Bs[i], nBs[i])
	Base = read_data(Bs[i], nBs[i])
	#print("Base = ", Base)
	Anom = []
	nameAs = ""
	totAnom = 0
	for n in range(len(nAs[i])):
		an = read_data(As[i][n], nAs[i][n])
		#nameAs = nameAs + As[i][n] + "_"
		nameAs = nameAs + As[i][n] + "_" + str(nAs[i][n]) + "_"
		totAnom = totAnom + nAs[i][n]
		#print("an = ", an)
		Anom.append(an[0])

	#print("Anom = ", Anom)

	print ("nameAs = ", nameAs)

	Data = []
	for v in Base:
		Data.append(v)
	for v in Anom:
		Data.append(v)

	path = "/home/antonioneme/Dropbox/anomDet_CDA/data/PDF/files/"
	name = "j_" + Bs[i] + "_anom_" + nameAs + ".csv"
	print("nm = ", name)

	#print("Data = ", Data)
	save_data(path + name, Data)

	cont = 1
	for k in nK:
		if k <= 2*totAnom + 1 and cont == 1:
		#if k <= 2*totAnom:
			cmd = "python3 /home/antonioneme/Dropbox/anomDet_CDA/progs/lof_cda_several_class.py  -i " + path + name + "  -k " + str(k) + " -u " + str(nBs[i]) + "  -o " + path + "lof_k_" + str(k) + "_" + name + "  -o2 " + path + "stats_lof_k_" + str(k) + "_" + name +  "  -o3 " + path + "class_lof_k_" + str(k) + "_" + name

			print("cmd = ", cmd)

			#cc = sys.stdin.read(1)
			os.system(cmd)

			#Stats = read_stats("/home/antonioneme/Dropbox/anomDet_CDA/data/PDF/files/class_lof_k_" + str(k) + "_" + name)
			Stats = read_stats(path + "stats_lof_k_" + str(k) + "_" + name)
			print ("Stats = ", Stats)
			save_stats(args.o, i, Bs[i], nBs[i], As[i], nAs[i], k, totAnom, Dists, Stats)
			cont = 0
			for dd in Dists:
				if Stats[dd][0] < 1.0 or Stats[dd][1] < 1.0:
					cont = 1
					break

