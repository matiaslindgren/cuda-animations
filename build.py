"""
Move index.html and source files to a build dir, while removing comments and all lines containing some strings such as "assert".
"""
import argparse
import os
import re
import shutil
import sys

assert_pattern = re.compile("assert")
comment_pattern = re.compile("^\s*?" + re.escape(r"//"))
index_html_filter_patterns = (
    re.compile(r"(?<=<script src=[\"'])src/"),
    re.compile(r"(?<=<link href=[\"'])(src|img)/"),
)

SOURCE_FILES = (
    ("main.js",     (assert_pattern, comment_pattern)),
    ("lib.js",      (assert_pattern, comment_pattern)),
    ("config.js",   (comment_pattern, )),
    ("kernels.js",  (assert_pattern, comment_pattern)),
    ("main.css",    ()),
)
SOURCE_FILES = tuple((os.path.join("src", path), pat) for path, pat in SOURCE_FILES)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", "-b",
        type=str,
        default=os.path.join(os.path.curdir, "build"),
        help="Path to build output directory")
    args = parser.parse_args()
    build_dir = args.build_dir
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.mkdir(build_dir)
    for src, drop_patterns in SOURCE_FILES:
        with open(src) as src_f:
            lines = src_f.readlines()
        dst = os.path.join(build_dir, os.path.basename(src))
        with open(dst, "w") as dst_f:
            for line in lines:
                if any(re.search(p, line) for p in drop_patterns):
                    continue
                dst_f.write(line)
    with open("index.html") as index_f:
        with open(os.path.join(build_dir, "index.html"), "w") as build_index_f:
            # Remove 'src/' from all link and script tag paths
            index_html = index_f.read()
            for pattern in index_html_filter_patterns:
                index_html = re.sub(pattern, '', index_html)
            build_index_f.write(index_html)
    shutil.copy(os.path.join("img", "chip.png"), os.path.join(build_dir, "chip.png"))
