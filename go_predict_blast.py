import argparse
import multiprocessing
import warnings

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

from lib.hmmer import hmmstats
from lib.hmmer import hmmscan


def main():
    """
    The go_predict_blast module takes an input blast TSV file containing sequences and a SVC model object and makes
    predictions about gene ontology based on the domain scores generated from a HMM domain model database
    """

    sh_parse = argparse.ArgumentParser(description="Predict the classification of a tsv file from cp-blast")
    sh_parse.add_argument("-f", "--file", dest="infile", help="Input sequence FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-c", "--column", dest="column", help="Sequence column NUMBER (0-index)", metavar="NUMBER",
                          required=True, type=int)
    sh_parse.add_argument("-o", "--out", dest="outfile", help="Output matrix FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-d", "--db", dest="database", help="Database FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-m", "--model", dest="modelfile", help="Model joblib FILE", metavar="FILE", required=True)
    sh_parse.add_argument("--cpu", dest="cores", help="Number of processor CORES to use", metavar="COREs", type=int,
                          default=1)

    sh_args = sh_parse.parse_args()

    go_predict_blast(sh_args.infile, sh_args.database, sh_args.modelfile, out_file=sh_args.outfile,
                     seq_column=sh_args.column, cores=sh_args.cores)


def go_predict_blast(infile_name, database_path, modelfile_name, out_file=None, seq_column=0, cores=2):
    svc_model_est = joblib.load(modelfile_name)

    hmmer_pool = multiprocessing.Pool(processes=cores, maxtasksperchild=1000)

    with open(infile_name, mode="rU") as in_fh:
        hmmer_imap = hmmer_pool.imap(PredictFromDomains(database_path, svc_model_est).hmmscan_predict,
                                     line_generator(in_fh, column=seq_column))

        with open(out_file, mode="w") as out_fh:
            for line, prediction, proba in hmmer_imap:
                print(line + "\t{}\t{}".format(prediction, proba), file=out_fh)


def line_generator(in_fh, column=0):
    for line in in_fh:

        line = line.strip()

        if line[0] == "#":
            continue

        line_tabs = line.split("\t")
        sequence = SeqRecord(Seq(line_tabs[column].strip(), alphabet=generic_protein),
                             id=line_tabs[1].strip(),
                             name=line_tabs[1].strip())

        yield (sequence, line)


class PredictFromDomains:
    def __init__(self, database, model, alpha=0.98):
        self.database = database
        self.domain_idx = hmmstats(database)
        self.model = model
        self.alpha = alpha

        print("Protein domain file parsed: {} domains detected".format(len(self.domain_idx)))

    def hmmscan_predict(self, data):
        sequence, line = data
        sparse_data, _ = hmmscan(sequence, self.database, self.domain_idx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predict_proba = self.model.predict_proba(normalize(sparse_data))[0]

        if predict_proba[1] > self.alpha:
            predict = True
        else:
            predict = False

        return line, predict, predict_proba[1]


if __name__ == '__main__':
    main()
