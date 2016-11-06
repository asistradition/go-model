import argparse

from sklearn.externals import joblib

from lib.hmmer import hmm_file_import
from lib.hmmer import hmmstats
from lib.go_parser import parse_gaf
from lib.svc import model_test


def main():
    sh_parse = argparse.ArgumentParser(description="Build a SVC model")
    sh_parse.add_argument("--hmmfile", dest="hmmfile", help="Input hmm prediction FILE", metavar="FILE", required=True)
    sh_parse.add_argument("--gofile", dest="gofile", help="Input Gene Ontology FILE", metavar="FILE", required=True)
    sh_parse.add_argument("--golist", dest="golist", help="Gene Ontology Terms to Model as a Text FILE", metavar="FILE",
                          required=True)
    sh_parse.add_argument("-d", "--db", dest="database", help="Database FILE", metavar="FILE", required=True)
    sh_parse.add_argument("-o", "--out", dest="outfile", help="Output model FILE", metavar="FILE", required=True)
    sh_parse.add_argument("--cpu", dest="cores", help="Number of processor CORES to use", metavar="COREs", type=int,
                          default=1)

    sh_args = sh_parse.parse_args()

    go_model_build(sh_args.hmmfile, sh_args.gofile, sh_args.golist, sh_args.outfile, sh_args.database,
                   core_num=sh_args.cores)


def go_model_build(hmmfile_path, gofile_path, golist_path, outfile_path, database_path, core_num=1):
    domain_col_idx = hmmstats(database_path + ".stats")
    print("Protein domain file parsed: {} domains detected".format(len(domain_col_idx)))

    protein_domain_matrix, protein_row_idx = hmm_file_import(hmmfile_path, domain_col_idx)
    print(
        "Protein hmm prediction file parsed: {} entries in {} x {} matrix".format(protein_domain_matrix.count_nonzero(),
                                                                                  protein_domain_matrix.shape[0],
                                                                                  protein_domain_matrix.shape[1]))

    go_list = []
    with open(golist_path, mode="rU") as go_list_fh:
        for line in go_list_fh:
            try:
                if line[0] == "#" or len(line) < 2:
                    continue
            except IndexError:
                continue

            go_list.append(line.strip())

    print("Gene Ontology list parsed: {} terms identified".format(len(go_list)))

    protein_go_matrix, go_col_idx = parse_gaf(gofile_path, protein_row_idx, go_list)

    print("Gene Ontology file parsed: {} entries in {} x {} matrix".format(protein_go_matrix.count_nonzero(),
                                                                           protein_go_matrix.shape[0],
                                                                           protein_go_matrix.shape[1]))

    for model_idx, model_name in enumerate(go_col_idx.keys()):
        print("Fitting Model #{} GO:{}".format(model_idx, model_name))
        best_model = model_test(protein_domain_matrix, protein_go_matrix[:, model_idx], cores=core_num)

        joblib.dump(best_model,"{}.{}.joblib".format(outfile_path, model_name), compress=True)

if __name__ == '__main__':
    main()
