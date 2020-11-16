"""
Prepares train, dev, test for a treebank

For example, do
  python -m stanza.utils.prepare_tokenizer_treebank TREEBANK
such as
  python -m stanza.utils.prepare_tokenizer_treebank UD_English-EWT

and it will prepare each of train, dev, test
"""

import glob
import os
import shutil
import sys

import stanza.utils.default_paths as default_paths
import stanza.utils.prepare_tokenizer_data as prepare_tokenizer_data
import stanza.utils.postprocess_vietnamese_tokenizer_data as postprocess_vietnamese_tokenizer_data

from stanza.models.common.constant import treebank_to_short_name

def prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, dataset):
    prepare_tokenizer_data.main([f"{udbase_dir}/{treebank}/{short_name}-ud-{dataset}.txt",
                                 f"{udbase_dir}/{treebank}/{short_name}-ud-{dataset}.conllu",
                                 "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                 "-m", f"{tokenizer_dir}/{short_name}-ud-{dataset}-mwt.json"])

    shutil.copyfile(f"{udbase_dir}/{treebank}/{short_name}-ud-{dataset}.conllu",
                    f"{tokenizer_dir}/{short_name}.{dataset}.gold.conllu")
    shutil.copyfile(f"{udbase_dir}/{treebank}/{short_name}-ud-{dataset}.txt",
                    f"{tokenizer_dir}/{short_name}.{dataset}.txt")

    if short_language == "vi":
        postprocess_vietnamese_tokenizer_data.main([f"{udbase_dir}/{treebank}/{short_name}-ud-{dataset}.txt",
                                                    "--char_level_pred", f"{tokenizer_dir}/{short_name}-ud-{dataset}.toklabels",
                                                    "-o", f"{tokenizer_dir}/{short_name}-ud-{dataset}.json"])

def process_treebank(treebank, paths):
    """
    Processes a single treebank into train, dev, test parts

    TODO
    Currently assumes it is always a UD treebank.  There are Thai
    treebanks which are not included in UD.

    Also, there is no specific mechanism for UD_Arabic-NYUAD or
    similar treebanks, which need integration with LDC datsets
    """
    udbase_dir = paths["UDBASE"]
    tokenizer_dir = paths["TOKENIZE_DATA_DIR"]

    short_name = treebank_to_short_name(treebank)
    short_language = short_name.split("_")[0]

    print("Preparing data for %s: %s, %s" % (treebank, short_name, short_language))

    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "train")
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "dev")
    prepare_ud_dataset(treebank, udbase_dir, tokenizer_dir, short_name, short_language, "test")

def get_ud_treebanks(udbase_dir):
    """
    Looks in udbase_dir for all the treebanks which have both train, dev, and test
    """
    treebanks = sorted(glob.glob(udbase_dir + "/UD_*"))
    treebanks = [t for t in treebanks
                 if (len(glob.glob(t + '/*train*')) > 0 and
                     len(glob.glob(t + '/*dev*')) > 0 and
                     len(glob.glob(t + '/*test*')) > 0)]
    treebanks = [os.path.split(t)[1] for t in treebanks]
    return treebanks


def main():
    if len(sys.argv) == 1:
        raise ValueError("Need to provide a treebank name")
    
    treebank = sys.argv[1]
    paths = default_paths.get_default_paths()
    if treebank.lower() == 'ud_all':
        treebanks = get_ud_treebanks(paths["UDBASE"])
        for t in treebanks:
            process_treebank(t, paths)
    else:
        process_treebank(treebank, paths)
    
if __name__ == '__main__':
    main()

