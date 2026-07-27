# -*- coding: utf-8 -*-
"""analysis/ -- promoted analyses: piloted in tests/, now supporting the paper.

Stage 2 of the analysis lifecycle:

    tests/  ->  analysis/  ->  figures_for_paper/
    pilot       promoted        published
                    |
                    +-> _archive/   (piloted, did not pan out)

Code lands here once it is worth depending on. Some of it is imported directly
by figures_for_paper/; some regenerates the results a paper figure is built
from; some is a completed analysis that supports the paper without having its
own figure. `analysis/README.md` records which is which per module -- check it
before moving or renaming anything here, because several of these modules are
load-bearing for figures despite not living in figures_for_paper/.

Anything under analysis/ is expected to keep working. Throwaway exploration
belongs in tests/.
"""
