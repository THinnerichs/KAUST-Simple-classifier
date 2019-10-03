import numpy as np

from sklearn.preprocessing import LabelEncoder

from repDNA import util, nac, nacutil
from repDNA.nac import Kmer
from repDNA.ac import DAC, DCC, TAC, TCC
from repDNA.psenac import PseDNC, PseKNC, PCPseDNC, PCPseTNC, SCPseDNC, SCPseTNC

import multiprocessing as mp

import time


def prepare_data_with_repDNA(include_acceptor=False,
                             include_donor=False,
                             save_file_name="dataset",
                             samples_per_file=20000,
                             start=0,
                             include_kmer=False,
                             include_DAC=False,
                             include_DCC=False,
                             include_TAC=False,
                             include_TCC=False,
                             include_PseDNC=False,
                             include_PseKNC=False,
                             include_PC_PseDNC=False,
                             include_PC_PseTNC=False,
                             include_SC_PseDNC=False,
                             include_SC_PseTNC=False):

    print("Reading data ...")

    # Prepare selected modes
    mode_list = []
    if include_acceptor:
        mode_list.append("acceptor")
    if include_donor:
        mode_list.append("donor")


    # Read data and perform transformation
    for b in mode_list:

        if include_kmer:
            x_dataset = []
            # kmer count occurences:
            kmer = Kmer(k=2, upto=True, normalize=True)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)
                my_time = time.time()

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(kmer.make_kmer_vec(seqs))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_kmer_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished Kmer data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))

        if include_DAC:
            # Calculate and store Dinuleotide-based auto covariance
            # Initialize datasets
            x_dataset = []
            dac = DAC(2)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(dac.make_dac_vec(seqs, all_property=True))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_dac_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished DAC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))

        if include_DCC:
            # Calculate and store Dinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []

            dcc = DCC(1)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(dcc.make_dcc_vec(seqs, all_property=True))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_dcc_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished DCC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))

        if include_TAC:
            # Calculate and store Trinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []

            tac = TAC(3)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(tac.make_tac_vec(seqs, all_property=True))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_tac_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished TAC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))

        if include_TCC:
            # Calculate and store Dinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []

            tcc = TCC(2)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]


                x_dataset.extend(tcc.make_tcc_vec(seqs, all_property=True))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_tcc_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished TCC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))

        if include_PseDNC:
            # Calculate and store Dinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []

            pseDNC = PseDNC(2)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(pseDNC.make_psednc_vec(seqs))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_pseDNC_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished PseDNC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))

        if include_PseKNC:
            # Calculate and store Dinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []
            pseKNC = PseKNC(k=2, lamada=1, w=0.05)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(pseKNC.make_pseknc_vec(seqs))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_pseKNC_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished pseKNC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))

        if include_PC_PseDNC:
            # Calculate and store Dinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []

            pc_psednc = PCPseDNC(lamada=2, w=0.05)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(pc_psednc.make_pcpsednc_vec(seqs, all_property=True))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_PC_PseDNC_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished PC-PseDNC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))


        if include_PC_PseTNC:
            # Calculate and store Dinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []

            pc_psetnc = PCPseTNC(lamada=2, w=0.05)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(pc_psetnc.make_pcpsetnc_vec(seqs, all_property=True))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_PC_PseTNC_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished PC-PseTNC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))


        if include_SC_PseDNC:
            # Calculate and store Dinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []

            sc_psednc = SCPseDNC(lamada=2, w=0.05)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(sc_psednc.make_scpsednc_vec(seqs, all_property=True))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_SC_PseDNC_" + save_file_name + ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished SC-PseDNC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))


        if include_SC_PseTNC:
            # Calculate and store Dinuleotide-based cross covariance
            # Initialize datasets
            x_dataset = []

            sc_psetnc = SCPseTNC(lamada=2, w=0.05)

            for a in ["negative", "positive"]:
                # Read data
                file_name = "../data/{}_{}.fa".format(a, b)
                print("Processing", file_name)

                seqs = util.get_data(open(file_name))[start:start+samples_per_file]

                x_dataset.extend(sc_psetnc.make_scpsetnc_vec(seqs, all_property=True))

            x_dataset = np.array(x_dataset, dtype=np.float)

            x_filename = "../data/x_SC_PseTNC_" + save_file_name+ ("_"+str(start) + "_start" if start != 0 else "") + "_" + str(samples_per_file) + "_samples.npy"
            # save dataset in numpy readable files
            np.save(file=x_filename, arr=x_dataset)

            print("Finished SC-PseTNC data.")
            print("Shape:", x_dataset.shape)
            print("Data saved in {}.".format(x_filename))

def acceptor_kmer_DAC():
    prepare_data_with_repDNA(include_acceptor=True,
                         include_donor=False,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                            start=100000,
                         include_kmer=True,
                         include_DAC=True,
                         include_DCC=False,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=False)
def acceptor_DCC():
    prepare_data_with_repDNA(include_acceptor=True,
                         include_donor=False,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=True,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=False)

def acceptor_PC_DNC():
    prepare_data_with_repDNA(include_acceptor=True,
                         include_donor=False,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=False,
                         include_PC_PseDNC=True,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=False)
def acceptor_PC_TNC():
    prepare_data_with_repDNA(include_acceptor=True,
                         include_donor=False,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=False,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=True,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=False)

def acceptor_SC_DNC():
    prepare_data_with_repDNA(include_acceptor=True,
                         include_donor=False,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=False,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=True,
                         include_SC_PseTNC=False)
def acceptor_SC_TNC():
    prepare_data_with_repDNA(include_acceptor=True,
                         include_donor=False,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=False,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=True)


def donor_kmer_DAC():
    prepare_data_with_repDNA(include_acceptor=False,
                         include_donor=True,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                         include_kmer=True,
                             start=100000,
                         include_DAC=True,
                         include_DCC=False,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=False)
def donor_DCC():
    prepare_data_with_repDNA(include_acceptor=False,
                         include_donor=True,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=True,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=False)

def donor_PC_DNC():
    prepare_data_with_repDNA(include_acceptor=False,
                         include_donor=True,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=False,
                         include_PC_PseDNC=True,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=False)
def donor_PC_TNC():
    prepare_data_with_repDNA(include_acceptor=False,
                         include_donor=True,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=False,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=True,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=False)

def donor_SC_DNC():
    prepare_data_with_repDNA(include_acceptor=False,
                         include_donor=True,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=False,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=True,
                         include_SC_PseTNC=False)
def donor_SC_TNC():
    prepare_data_with_repDNA(include_acceptor=False,
                         include_donor=True,
                         save_file_name="acceptor_data",
                         samples_per_file=10000,
                             start=100000,
                         include_kmer=False,
                         include_DAC=False,
                         include_DCC=False,
                         include_PC_PseDNC=False,
                         include_PC_PseTNC=False,
                         include_SC_PseDNC=False,
                         include_SC_PseTNC=True)





if __name__ == '__main__':

    '''
    prepare_data_with_repDNA(include_acceptor=True,
                             include_donor=False,
                             save_file_name="acceptor_data",
                             samples_per_file=100000,
                             include_kmer=False,
                             include_DAC=False,
                             include_DCC=True,
                             include_PC_PseDNC=False,
                             include_PC_PseTNC=False,
                             include_SC_PseDNC=False,
                             include_SC_PseTNC=False)


    prepare_data_with_repDNA(include_acceptor=False,
                             include_donor=True,
                             save_file_name="donor_data",
                             samples_per_file=100000,
                             include_kmer=False,
                             include_DAC=False,
                             include_DCC=False,
                             include_PC_PseDNC=False,
                             include_PC_PseTNC=False,
                             include_SC_PseDNC=False,
                             include_SC_PseTNC=False)
    '''

    jobs = [acceptor_kmer_DAC, acceptor_DCC, acceptor_PC_DNC, acceptor_PC_TNC, acceptor_SC_DNC, acceptor_SC_TNC]
    jobs.extend([donor_kmer_DAC, donor_DCC, donor_PC_DNC, donor_PC_TNC, donor_SC_DNC, donor_SC_TNC])

    for job in jobs:
        p = mp.Process(target=job)
        p.start()
