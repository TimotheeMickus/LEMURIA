import filecmp
import itertools as it
import os

def rm_dups(sequential_only=True, dataset_path='/home/airc/Downloads/rgb_dataset'):
    CWD = os.getcwd()
    os.chdir(dataset_path)
    files = set(filter(os.path.isfile, os.listdir('.')))
    delme = set()

    files = sorted(list(files), key=lambda f:os.stat(f).st_ctime)
    print(len(files), 'files found')
    paired_files = zip(files, files[1:]) \
        if sequential_only \
        else it.combinations(files, 2)

    for f1, f2 in paired_files:
        if not os.stat(f1).st_size:
            delme.add(f1)
        if not os.stat(f2).st_size:
            delme.add(f2)
        if f1 in delme and f2 in delme:
            print('skipping', f1, 'and', f2)
            continue
        if filecmp.cmp(f1,f2,shallow=False):
            delme.add(f1)
            delme.add(f2)

    if len(delme):
        print("The following %i files were found to be duplicates and will be removed:" % len(delme))
        print("\t", *sorted(list(delme), key=lambda f:int(f.split('_')[0])))
        for f in delme:
            os.remove(f)

    else:
        print("No duplicates found")
    os.chdir(CWD)
    return delme

if __name__ == "__main__":
    rm_dups()
