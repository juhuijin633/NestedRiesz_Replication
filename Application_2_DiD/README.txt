Application_2_DiD
=================

Minimum-wage DiD application.

Layout
------
  code/
    RUN.py
    1_fetch_data.py
    2_calc_estimates.py
    3_tables_figs.py
    utils/
      load_minwage_data.py   CSV → Y1, Y2, D, Z, X1, X2 tensors
      estimateDiD_OLS.py, dynamicRiesz*.py
  data/raw/
  results/intermediate/
  results/

Run
---
  cd Application_2_DiD/code
  python RUN.py
  python RUN.py --skip-fetch --skip-estimates   # figures only

Intermediate CSVs use a `method_id` column (e.g. static_ols, dynamic_auto_rf).

Years: 2004–2007 (pre-year 2003, treatment year 2004).
