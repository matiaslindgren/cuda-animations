"""
Move source files to a build dir, while removing comments and all lines containing some strings such as "assert".
"""
import argparse
import os
import re

assert_pattern = re.compile("assert")
comment_pattern = re.compile("^\s*?" + re.escape(r"//"))

SOURCE_FILES = (
    ("src/main.js",     (assert_pattern, comment_pattern)),
    ("src/lib.js",      (assert_pattern, comment_pattern)),
    ("src/config.js",   (comment_pattern, )),
    ("src/kernels.js",  (assert_pattern, comment_pattern)),
    ("src/main.css",    ()),
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
    for src, drop_patterns in SOURCE_FILES:
        with open(src) as f:
            lines = f.readlines()
        dst = os.path.join(args.build_dir, os.path.basename(src))
        with open(dst, "w") as f:
            for line in lines:
                if any(re.search(p, line) for p in drop_patterns):
                    continue
                f.write(line)
