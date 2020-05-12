single_pass: main.tex
	pdflatex main.tex

references: main.tex references.bib
	pdflatex main.tex
	biber main
	pdflatex main.tex
	pdflatex main.tex

.PHONY:	clean
clean:
	rm -f main.aux main.log main.nav main.out main.pdf main.snm main.toc

