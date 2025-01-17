import itertools


def convert():
    """
    This function converts the raw data files in .fa format to Biopython's SeqIO readable FASTA files.
    The files are saved in the same directory.

    :return:
    """
    for a,b in itertools.product(["negative", "positive"], ["acceptor", "donor"]):
        # Read data
        input_file_name = "../data/{}_DNA_seqs_{}_at.fa".format(a,b)
        output_file_name = "../data/{}_{}.fa".format(a,b)
        print("Processing", input_file_name)
        with open(file=input_file_name, mode='r') as infile:
            counter = 0
            with open(file=output_file_name, mode='w') as outfile:
                for input_line in infile:
                    outfile.write(">" + str(counter) +"\n")
                    outfile.write(input_line)
                    counter += 1

if __name__ == '__main__':
    convert()
