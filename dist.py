"""
Move source files to a build dir, while removing all lines containing some strings such as "assert".
"""
import argparse
import os

SOURCE_FILES = (
    ("src/main.js", "assert"),
    ("src/lib.js", "assert"),
    ("src/config.js", ''),
    ("src/kernels.js", ''),
    ("src/main.css", ''),
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build_dir", "-b",
        type=str,
        default=os.path.join(os.path.curdir, "build"),
        help="Path to build output directory")
    args = parser.parse_args()
    if not os.path.exists(args.build_dir):
        os.mkdir(args.build_dir)
    for src, drop_string in SOURCE_FILES:
        with open(src) as f:
            lines = f.readlines()
        dst = os.path.join(args.build_dir, os.path.basename(src))
        with open(dst, "w") as f:
            for line in lines:
                if drop_string and drop_string in line:
                    continue
                f.write(line)
