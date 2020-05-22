single_pass: main.tex
	lualatex main.tex

references: main.tex references.bib
	lualatex main.tex
	biber main
	lualatex main.tex
	lualatex main.tex

.PHONY:	clean
clean:
	rm -f main.aux main.bbl main.bcf main.blg main.log main.run.xml main.out main.run.xml main.tex.bbl main.tex.blg main.ttt missfont.log q.log main.pdf
