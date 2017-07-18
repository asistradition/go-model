import subprocess
import io

from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

GAF_FILE_FORMAT = {"PROT_ACC": 1,
                   "GO_TERM": 3}

TSV_FILE_FORMAT = {"PROT_ACC": 0,
                   "GO_TERM": 1}


def parse_gaf(gaf_path, row_idx, terms_to_keep, file_type="gaf"):
    """
    parse_gaf parses the gene ontology consortium GAF files for protein accession and go term

    Required Arguments:

    :param gaf_path: str
        Path to the GAF file to parse
    :param row_idx: dict
        A dict of row numbers keyed by row name
    :param terms_to_keep: list
        A list of GO terms to retain after parsing

    Keyword Arguments:

    :param file_type: str
        Pass in gaf for a gaf file or tsv for a tsv file

    Returns:

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

    if file_type == "gaf":
        file_format = GAF_FILE_FORMAT
    elif file_type == "tsv":
        file_format = TSV_FILE_FORMAT
    else:
        raise AttributeError("File_type {} Unknown".format(file_type))

    go_data = lil_matrix((len(row_idx), len(terms_to_keep)), dtype=bool)

    for go_id in terms_to_keep:
        #Run through the GAF file to get the GO terms that are present
        #grep is just so much faster then doing this entirely in python

        grep_cmd = ["grep", str(go_id), gaf_path]
        grep_proc = subprocess.Popen(grep_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     universal_newlines=True)
        matching_go_lines = io.StringIO(grep_proc.communicate()[0].strip())

        for line in matching_go_lines:

            # Get the accession and go term
            try:
                if line[0] == "!" or len(line) < 2:
                    continue

                line_tabs = line.split()
                line_acc = line_tabs[file_format["PROT_ACC"]].strip()
                line_go = line_tabs[file_format["GO_TERM"]].split(":")[1]
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
