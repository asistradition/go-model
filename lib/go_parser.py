from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix


def parse_gaf(gaf_path, row_idx, terms_to_keep):
    """
    parse_gaf parses the gene ontology consortium GAF files for protein accession and go term

    :param gaf_path: str
        Path to the GAF file to parse
    :param row_idx: dict
        A dict of row numbers keyed by row name
    :param terms_to_keep: list
        A list of GO terms to retain after parsing

    :return go_data: scipy.sparse.csr_matrix
        The GO array as a sparse matrix
    :return go_idx: dict
        A dict of column numbers keyed by GO term
    """


    # Build a term index from the list of terms to keep
    go_idx = {}

    for idx, go_term in enumerate(terms_to_keep):
        go_idx[go_term] = idx

    go_data = lil_matrix((len(row_idx), len(terms_to_keep)), dtype=bool)

    with open(gaf_path, mode="rU") as gaf_fh:
        # Run through the GAF file to get the GO terms that are present
        for line in gaf_fh:
            if line[0] == "!" or len(line) < 2:
                continue

            try:
                line_tabs = line.split()
                line_acc = line_tabs[1].strip()
                line_go = line_tabs[3].split(":")[1]
            except IndexError:
                continue

            try:
                if go_idx[line_go] == 0 or row_idx[line_acc] == 0:
                    pass
            except KeyError:
                continue

            try:
                go_data[row_idx[line_acc], go_idx[line_go]] = True
            except KeyError:
                continue

    go_data = csr_matrix(go_data, dtype=bool)

    return go_data, go_idx


def parse_obo(obo_path):
    go_dict = {}

    with open(obo_path, mode="rU") as obo_fh:
        for line in obo_fh:
            line_split = line.split(":")

            try:
                if line_split[0] == "id" and line_split[2] not in go_dict:
                    next_id = len(go_dict)
                    go_dict[line_split] = next_id
            except IndexError:
                continue

    return go_dict
