.PHONY: all book html epub pdf check snippets images clean

all: check book

book: snippets
	quarto render

html: snippets
	quarto render --to html

epub: snippets
	quarto render --to epub

pdf: snippets
	quarto render --to pdf

check: snippets
	python3 src/python/run_book_code.py

snippets:
	python3 tools/refresh_code_snippets.py

images:
	bash src/matplotlib/render_all.sh

clean:
	rm -rf _book .quarto
