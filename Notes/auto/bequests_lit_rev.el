(TeX-add-style-hook
 "bequests_lit_rev"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("biblatex" "backend=bibtex" "style=authoryear")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "./Equations/Abel_1985"
    "./Equations/Cagetti_2008"
    "./Equations/Barro_1974"
    "./Equations/Cagetti_2006"
    "./Equations/Carroll_1998"
    "./Equations/Cagetti_2003"
    "./Equations/Gourinchas_2002"
    "./Equations/DeNardi_2004"
    "./Equations/Dynan_2004"
    "./Equations/DeNardi_2016"
    "./Equations/Saez_2017"
    "./Equations/Straub_2019"
    "article"
    "art10"
    "color"
    "hyperref"
    "amsmath"
    "amsfonts"
    "dirtytalk"
    "inputenc"
    "biblatex")
   (TeX-add-symbols
    '("doi" 1)
    '("doilink" 1))
   (LaTeX-add-bibliographies
    "BeqRefs"))
 :latex)

