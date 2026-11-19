# =====================================================================
#  Build the movement-recognition FFT feature-pipeline figure.
#
#  Produces a single combined figure (three stacked stages: raw recording,
#  overlapping sliding windows, per-movement FFT power) as PNG + SVG.
#
#  Targets:
#    figures  -- run the Python script to generate the figure.
#    clean    -- remove generated artefacts.
#
#  DATA points at the directory of `collect` CSV recordings; override it
#  on the command line, e.g.  make figures DATA=/path/to/data
# =====================================================================

PYTHON ?= python3
DATA   ?= ../../data

FIG_DIR   := figures
FIG_STAMP := $(FIG_DIR)/.stamp

.PHONY: all figures clean
all: figures

figures: $(FIG_STAMP)
$(FIG_STAMP): generate_figures.py
	$(PYTHON) $< --data $(DATA) --outdir $(FIG_DIR)
	@touch $@

clean:
	rm -rf $(FIG_DIR)
