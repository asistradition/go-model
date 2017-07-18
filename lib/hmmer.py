import subprocess
import tempfile
import math
import numpy
import os
import io

from Bio import SeqIO
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

LINE_SCORE_IDX = 5
LINE_QUERY_IDX = 2
LINE_DOMAIN_IDX = 1

class HmmscanProcess:
    """
    The HmmscanProcess class facilitates multiprocesing through Pool
    It's only appropriate for saving to a file it throws away the data matrix
    """

    def __init__(self, database_path, domain_idx, core_num=2):
        self.database_path = database_path
        self.domain_idx = domain_idx
        self.core_num = core_num

    def scan(self, data_package):
        data, loop_num, slice_start, slice_stop = data_package

        fake_file_handle = io.StringIO()
        hmmscan(data, self.database_path, self.domain_idx, output_file_handle=fake_file_handle, core_num=self.core_num)
        return fake_file_handle, loop_num, slice_start, slice_stop


def hmmscan(seq_records, database_path, domain_idx, output_file_handle=None, core_num=1):
    """
    hmmscan is a wrapper for command line hmmscan from the hmmer package. It takes a list of SeqRecords and searches a
    hmm database. It takes the output file and parses it into a sparse matrix

    Required Arguments:

    :param seq_records: [Bio.SeqRecord]
        A list of SeqRecords to process
    :param database_path: str
        Path to the hmmer database
    :param domain_idx: dict
        An index dict keyed by domain accession containing the column key

    Keyword Arguments:

    :param output_file_handle: file object
        If set, will save protein_name, domain, value to the TSV file
    :param core_num: int
        The number of cores to pass to hmmscan

    Returns:

    :return hmmer_scores: scipy.sparse.csr_matrix
        A 2d sparse csr matrix with columns of protein domains and rows of query names
    :return query_idx: dict
        An index dict keyed by protein name containing the row key
    """

    # Make temp files and save the queries to them as a fasta file
    fasta_temp = tempfile.mkstemp(suffix=".fasta")
    hmmer_temp = tempfile.mkstemp(suffix=".hmmer")
    with open(fasta_temp[0], mode="w") as seq_file:
        SeqIO.write(seq_records, seq_file, format="fasta")

    # Run hmmscan on the fasta file and throw away the
    hmmscan_cmd = ["hmmscan", "--cpu", str(core_num), "--tblout", hmmer_temp[1], database_path, fasta_temp[1]]
    subprocess.call(hmmscan_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    query_idx = {}
    hmmer_scores = numpy.zeros((len(seq_records), len(domain_idx)), dtype=numpy.int16)

    with open(hmmer_temp[0], mode="rU") as hmmer_file:
        for line in hmmer_file:
            if line[0] == "#" or len(line) < 2:
                continue
            line_tabs = line.strip().split()

            try:
                line_score = math.ceil(float(line_tabs[LINE_SCORE_IDX]) * 10)
                line_query = line_tabs[LINE_QUERY_IDX].strip()
                line_domain = line_tabs[LINE_DOMAIN_IDX].strip()
            except IndexError:
                continue

            if line_query not in query_idx:
                query_len = len(query_idx)
                query_idx[line_query] = query_len

            hmmer_scores[query_idx[line_query], domain_idx[line_domain]] = line_score

            if output_file_handle is not None:
                print("{}\t{}\t{}".format(line_query, line_domain, line_score), file=output_file_handle)

    os.remove(fasta_temp[1])
    os.remove(hmmer_temp[1])

    sparse_scores = csr_matrix(hmmer_scores[0:len(query_idx), :], dtype=numpy.int16)

    return sparse_scores, query_idx


def hmmstats(database_path):
    """
    hmmstats retrieves a list of domain accessions from the hmm database file

    :param database_path: str
        Path to the hmm database file

    :return hmm_name: dict
        A dict of column index names keyed by the domain accession
    """

    # Check to make sure that the database exists in a hmmbuild packed format
    if not os.path.isfile(database_path + ".h3p"):
        raise FileNotFoundError(database_path + ".h3p")

    # Run hmmstat on the hmm database and get the stdout
    hmmstat_output = subprocess.Popen(["hmmstat", database_path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      universal_newlines=True).communicate()[0]

    hmm_name = {}

    # Iterate through the hmmstat output by line and save an index dict
    for line in hmmstat_output.splitlines():
        try:
            if "#" == line[0]:
                continue

            hmm_name_len = len(hmm_name)
            hmm_name[line.split()[2].strip()] = hmm_name_len
        except IndexError:
            continue

    return hmm_name


def hmm_file_import(data_file, domain_idx):
    """
    hmm_file_import takes a data file in protein_name, domain, score format and turns it into a sparse matrix and
    a row index keyed by protein name

    Required Arguments:

    :param data_file: str
        Path to the data flatfile
    :param domain_idx: dict
        Domain index keyed by domain name to column number

    Returns:

    :return sparse_data: scipy.sparse.csr_matrix
        2d sparse matrix of hmm scores with protein name rows and domain columns
    :return row_name_idx: dict
        Row index keyed by protein name to row number
    """

    with open(data_file, mode="rU") as data_fh:

        # Run through the input file and save unique protein accession IDs into a index dict

        row_name_idx = {}
        for line in data_fh:
            try:
                if line[0] == "#" or len(line) < 2:
                    continue
                line_name = line.strip().split()[0]
            except IndexError:
                continue

            if line_name not in row_name_idx:
                next_idx = len(row_name_idx)
                row_name_idx[line_name] = next_idx

        data_fh.seek(0)

        # Run through the input file again and save the domain scores into a sparse lil_matrix
        # Building to a dense numpy array is faster but the memory usage on large data sets is extreme

        hmmer_data = lil_matrix((len(row_name_idx), len(domain_idx)), dtype=numpy.int16)
        for line in data_fh:
            try:
                if line[0] == "#" or len(line) < 2:
                    continue

                line_tabs = line.strip().split()
                line_name = line_tabs[0]
                line_domain = line_tabs[1]
                line_score = int(line_tabs[2])
            except IndexError:
                continue

            hmmer_data[row_name_idx[line_name], domain_idx[line_domain]] = line_score

    # Convert the lil_matrix to a csr_matrix and return it
    return csr_matrix(hmmer_data, dtype=numpy.int16), row_name_idx
