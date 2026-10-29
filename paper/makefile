# Optimizing AoII on real-world IoT sensor datasets. IEEEtran conference paper, biblatex (biber) bibliography in paper.bib.
LATEX = pdflatex -interaction=nonstopmode -halt-on-error
FIGURES = $(wildcard figure/*.tex figure/*/*.pdf figure/*/*.png figure/*/*.jpg figure/*/*.jpeg)

all: paper

paper: paper.pdf
slide: slide.pdf

paper.pdf: paper.tex paper.sty paper.bib $(FIGURES)
	$(LATEX) paper
	biber -quiet paper
	$(LATEX) paper
	$(LATEX) paper
	@echo "--- pages: $$(pdfinfo paper.pdf | awk '/^Pages/{print $$2}')"
	@echo "--- overfull boxes: $$(grep -c Overfull paper.log)"
	@echo "--- undefined references: $$(grep -c 'undefined' paper.log)"

slide.pdf: slide.tex slide.sty paper.bib $(FIGURES)
	$(LATEX) slide
	biber -quiet slide
	$(LATEX) slide
	$(LATEX) slide
	@echo "--- slides: $$(pdfinfo slide.pdf | awk '/^Pages/{print $$2}')"
	@echo "--- overfull boxes: $$(grep -c Overfull slide.log)"

pages:
	pdfinfo paper.pdf | grep Pages
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.acn *.bcf *.glo *.ist *.nav *.run.xml *.snm *.fls *.fdb_latexmk *.synctex.gz paper.pdf slide.pdf
view:
	zathura paper.pdf
slideview:
	zathura slide.pdf

.PHONY: all paper slide pages clean view slideview
