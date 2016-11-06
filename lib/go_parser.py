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

    # Build the gene ontology data into a sparse matrix
    # Dense is faster but pretty memory intensive

    go_data = lil_matrix((len(row_idx), len(terms_to_keep)), dtype=bool)
    with open(gaf_path, mode="rU") as gaf_fh:
        # Run through the GAF file to get the GO terms that are present
        for line in gaf_fh:

            # Get the accession and go term
            try:
                if line[0] == "!" or len(line) < 2:
                    continue

                line_tabs = line.split()
                line_acc = line_tabs[1].strip()
                line_go = line_tabs[3].split(":")[1]
            except IndexError:
                continue

            # If they're not in the indexes move on
            try:
                if go_idx[line_go] == 0 or row_idx[line_acc] == 0:
                    pass
            except KeyError:
                continue

            # Set the matrix value if they're present
            try:
                go_data[row_idx[line_acc], go_idx[line_go]] = True
            except KeyError:
                continue

    # Convert the lil_matrix into a csr_matrix and return it
    go_data = csr_matrix(go_data, dtype=bool)

    return go_data, go_idx
