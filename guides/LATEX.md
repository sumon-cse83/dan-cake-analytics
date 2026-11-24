# Compiling the LaTeX report

## Option A: Overleaf (easiest, free, nothing to install)

1. Create a free account at https://www.overleaf.com.
2. Use `report_overleaf.zip` in the project folder (the `report` folder is self-contained), or zip the `report`
   folder yourself.
3. Overleaf → **New Project → Upload Project** → select the zip.
4. Menu → set **Main document** = `main.tex`, compiler = **pdfLaTeX**. Click **Recompile**.
5. Download the PDF and copy it to `docs/report.pdf` for the website.

## Option B: on this PC

Install MiKTeX (https://miktex.org) or TeX Live, then:

```powershell
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Before submission

* Edit your **name, student ID, supervisor, department, university and date** at the top of `report/main.tex`
  (the `\newcommand{\StudentName}{…}` block). Replace the `YOUR-USER` links once deployed.
* Read every chapter and adapt the wording to your own voice. You must be able to explain every sentence in
  your viva or presentation.
* Check each reference in `references.bib` against your library. All are real, published works.
* Run the report through your university's similarity checker (e.g. Turnitin). The text was written
  originally for this project and every source is cited. Similarity tools usually match standard phrases,
  reference lists and the proposal text you wrote earlier, so review the matches report itself rather than
  only the percentage.
* Follow your university's rules on declaring AI or software assistance.

## How results stay correct

`report/generated/numbers.tex` (e.g. `\CAGRSales`) and `report/tables/*.tex` are written by
`python run_pipeline.py`. After re-running the pipeline, recompile and every number updates. Never type
results by hand.
