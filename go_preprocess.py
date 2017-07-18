import argparse
import math
import multiprocessing

from Bio import SeqIO
from Bio.Alphabet import generic_protein

from lib.hmmer import hmmstats
from lib.hmmer import HmmscanProcess


def main():
    """
    The go_preprocess module takes an input sequence file and turns it into a HMM prediction flatfile using a HMM
    model database.
    """

    sh_parse = argparse.ArgumentParser(description="Preprocess a fasta file into a HMM prediction flatfile")
    sh_parse.add_argument("-f", "--file", dest="infile", help="Input sequence FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-o", "--out", dest="outfile", help="Output matrix FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-d", "--db", dest="database", help="Database FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-t", "--type", dest="intype", help="Input filename TYPE", metavar="TYPE", default="fasta")
    sh_parse.add_argument("--cpu", dest="cores", help="Number of processor CORES to use", metavar="COREs", type=int,
                          default=1)

    sh_args = sh_parse.parse_args()

    go_preprocess(sh_args.infile, sh_args.outfile, sh_args.database, in_type=sh_args.intype, core_num=sh_args.cores)


def go_preprocess(in_file, out_file, database_path, in_type="fasta", batch_size=200, core_num=1):
    """
    go_preprocess is the worker function to take an input sequence file and score protein domains based on a HMM model
    database

    Required Arguments:

    :param in_file: str
        Location of the input sequence file
    :param out_file: str
        Location of the output protein domain score flatfile
    :param database_path:
        Location of the hmm model database files

    Keyword Arguments:

    :param in_type: str
        Format of the input sequence file. Takes anything SeqIO does.
    :param batch_size: int
        Break jobs into batch pieces and multiprocess.pool them through hmmscan. Faster then giving hmmscan more cores
    :param core_num: int
        Number of cores to use for parallelization
    """

    # Parse the input sequence file into a list of seq_records
    with open(in_file, mode="rU") as in_fh:
        input_data = list(SeqIO.parse(in_fh, in_type, alphabet=generic_protein))

    print("File Read Complete: {} Sequences Detected".format(len(input_data)))

    # Parse the hmm model database stats
    try:
        domain_idx = hmmstats(database_path)
        print("Identified {} HMM models in {}".format(len(domain_idx), database_path))
    except FileNotFoundError:
        print("Database file not found")
        exit(0)

    # Instantiate the HmmscanProcess class, which is a wrapper for hmmscan to facilitate imap through
    # multiprocessing.Pool
    hmmscan_process = HmmscanProcess(database_path, domain_idx, core_num=2)

    # Build a generator to create data slices to pass to multiprocessing.Pool
    def _data_slice_gen(data, size):
        for looper in range(math.ceil(len(data) / size)):
            slice_start = looper * size
            slice_stop = slice_start + size

            if slice_stop > len(data):
                slice_stop = len(data)

            yield (data[slice_start:slice_stop], looper, slice_start, slice_stop)

    with open(out_file, mode="w") as out_fh:

        process_pool = multiprocessing.Pool(processes=core_num, maxtasksperchild=100)

        for string_fh, loop_num, start, stop in process_pool.imap(hmmscan_process.scan,
                                                                  _data_slice_gen(input_data, batch_size)):
            print(string_fh.getvalue(), file=out_fh)
            print("HMMSCAN #{} completed on records {}-{}".format(loop_num, start, stop))
            string_fh.close()


if __name__ == '__main__':
    main()
