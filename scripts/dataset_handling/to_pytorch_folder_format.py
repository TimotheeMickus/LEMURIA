import pathlib
import shutil

def main(input_path, output_path, do_mv=False):
    move = shutil.move if do_mv else shutil.copy
    for file in input_path.glob("**/*.png"):
        label = file.name.split("_", 1)[1].split(".")[0]
        tgt_dir = output_path / label
        tgt_dir.mkdir(exist_ok=True)
        move(file, tgt_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=pathlib.Path)
    parser.add_argument("output_path", type=pathlib.Path)
    parser.add_argument("--do_mv", action="store_true")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.do_mv)
