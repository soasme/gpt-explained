.PHONY: all book html epub pdf check snippets images publish clean

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
	bash src/dot/render_all.sh

publish: book
	git -C _book init
	git -C _book checkout -B gh-pages
	git -C _book remote remove origin 2>/dev/null || true
	git -C _book remote add origin git@github.com:soasme/gpt-explained.git
	git -C _book add -A
	git -C _book commit -m "Publish book"
	git -C _book push origin gh-pages -f

clean:
	rm -rf _book .quarto
