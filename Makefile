single_pass: main.tex
	pdflatex main.tex

references: main.tex references.bib
	pdflatex main.tex
	biber main
	pdflatex main.tex
	pdflatex main.tex

.PHONY:	clean
clean:
	rm -f main.aux main.bbl main.bcf main.blg main.log main.run.xml
