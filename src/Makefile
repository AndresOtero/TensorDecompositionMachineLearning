propuesta.pdf:tesis.tex tesis.bbl
	pdflatex propuesta
	pdflatex propuesta

propuesta.bbl: tesis.bib
	bibtex propuesta

propuesta.aux:tesis.tex
	pdflatex propuesta

propuesta.bib:tesis.bib
	cat bibliography.bib|sed -e 's/^\([ \t]*\)[mM]onth = "\([a-z]..\)"/\1month = \2/'>propuesta.bib

build:clean propuesta.pdf

clean:
	rm -f propuesta.aux propuesta.bbl propuesta.pdf propuesta.bib *.blg *.dvi *.lo? *.out *.to? *.pdf

.PHONY:clean build
