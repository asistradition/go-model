import argparse

from Bio import SeqIO
from Bio.Alphabet import generic_protein
from sklearn.externals import joblib

from lib.hmmer import hmmstats
from lib.hmmer import hmmscan


def main():
    sh_parse = argparse.ArgumentParser(description="Preprocess a fasta file into a HMM prediction flatfile")
    sh_parse.add_argument("-f", "--file", dest="infile", help="Input sequence FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-o", "--out", dest="outfile", help="Output matrix FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-d", "--db", dest="database", help="Database FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-m", "--model", dest="modelfile", help="Model joblib FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-t", "--type", dest="intype", help="Input filename TYPE", metavar="TYPE", default="fasta")
    sh_parse.add_argument("--cpu", dest="cores", help="Number of processor CORES to use", metavar="COREs", type=int,
                          default=1)

    sh_args = sh_parse.parse_args()

    go_predict(sh_args.infile, sh_args.outfile, sh_args.database, sh_args.modelfile, in_type=sh_args.intype,
               core_num=sh_args.cores)


def go_predict(in_file, out_file, database_path, modelfile_path, in_type="fasta", core_num=1):
    svc_model_est = joblib.load(modelfile_path)
    domain_col_idx = hmmstats(database_path + ".stats")

    with open(in_file, mode="rU") as in_fh:
        seq_record_list = list(SeqIO.parse(in_file, format=in_type, alphabet=generic_protein))

    print("File Read Complete: {} Sequences Detected".format(len(seq_record_list)))

    domain_data, protein_row_idx = hmmscan(seq_record_list, database_path, domain_col_idx, core_num=core_num)

    print("Protein hmm scan complete: {} entries in {} x {} matrix".format(domain_data.count_nonzero(),
                                                                           domain_data.shape[0],
                                                                           domain_data.shape[1]))

    predictions = svc_model_est.predict_proba(domain_data)

    print("Protein GO prediction complete: {} x {} predictions".format(predictions.shape[0], predictions.shape[1]))
    true_probabilities = predictions[:, 1].ravel()

    with open(out_file, mode="w") as out_fh:
        for protein_name in protein_row_idx.keys():
            print("{}\t{}".format(protein_name, true_probabilities[protein_row_idx[protein_name]]), file=out_fh)


if __name__ == '__main__':
    main()
