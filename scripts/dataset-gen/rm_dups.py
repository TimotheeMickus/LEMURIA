import filecmp
import itertools as it
import os

def rm_dups():
    DATASET_PATH = '../Downloads/rgb_dataset'
    CWD = os.getcwd()
    os.chdir(DATASET_PATH)
    files = set(filter(os.path.isfile, os.listdir('.')))
    delme = set()

    for f1, f2 in it.combinations(files, 2):
        if f1 in delme and f2 in delme: continue
        if filecmp.cmp(f1,f2,shallow=False):
            delme.add(f1)
            delme.add(f2)

    if len(delme):
        print("The following files were found to be duplicates and will be removed:")
        print("\t", *delme)
        for f in delme:
            os.remove(f)

    else:
        print("No duplicates found")
    os.chdir(CWD)
    return delme

if __name__ == "__main__":
    rm_dups()
