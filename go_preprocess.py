import argparse
import math
import io
import multiprocessing

from Bio import SeqIO
from Bio.Alphabet import generic_protein

from lib.hmmer import hmmstats
from lib.hmmer import hmmscan


def main():
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
    with open(in_file, mode="rU") as in_fh:
        input_data = list(SeqIO.parse(in_fh, in_type, alphabet=generic_protein))

    print("File Read Complete: {} Sequences Detected".format(len(input_data)))

    try:
        domain_idx = hmmstats(database_path + ".stats")
    except FileNotFoundError:
        print("Database stats file not found")
        exit(0)

    hmmscan_process = HmmscanProcess(database_path, domain_idx, core_num=2)

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


class HmmscanProcess:
    def __init__(self, database_path, domain_idx, core_num=2):
        self.database_path = database_path
        self.domain_idx = domain_idx
        self.core_num = core_num

    def scan(self, data_package):
        data, loop_num, slice_start, slice_stop = data_package

        fake_file_handle = io.StringIO()
        hmmscan(data, self.database_path, self.domain_idx, output_file_handle=fake_file_handle, core_num=self.core_num)
        return fake_file_handle, loop_num, slice_start, slice_stop


if __name__ == '__main__':
    main()
