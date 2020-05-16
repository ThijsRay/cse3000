single_pass: main.tex
	lualatex main.tex

references: main.tex references.bib
	lualatex main.tex
	biber main
	lualatex main.tex
	lualatex main.tex

.PHONY:	clean
clean:
	rm -f main.aux main.bbl main.bcf main.blg main.log main.run.xml main.pdf
