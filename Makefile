SHELL=/bin/bash
OUTDIR=build
SRC_FILES={kernels,config,lib,main}.js

.PHONY: dirs clean rm-js-asserts minify-js all

dirs:
	mkdir -pv $(OUTDIR)

clean:
	rm -rv $(OUTDIR)

rm-js-asserts:
	python3 build.py --build-dir $(OUTDIR)

minify-js:
	cat $(OUTDIR)/$(SRC_FILES) | python3 -m rjsmin > $(OUTDIR)/main.min.js && rm -v $(OUTDIR)/$(SRC_FILES)

all: dirs rm-js-asserts minify-js
