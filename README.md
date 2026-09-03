# Machine learning vs structural gravity on EU imports of textile trimmings

Code and results for my MSc dissertation at Gisma University of Applied Sciences (Data Science, AI and Digital Business):

*Machine Learning versus Structural Gravity: Predicting and Explaining EU Import Flows of Textile Trimmings, with an Application to Brazilian Suppliers (2015-2025)*

The idea: take EU imports of textile trimmings (lace, ribbons, braids, embroidery, buttons and zippers, HS headings 5804, 5806, 5808, 5810, 9606, 9607) at HS6 level and check whether machine learning models actually beat a gravity model estimated with PPML when both get exactly the same data. Then use both models to see where Brazil sells less to the EU than you would expect from its fundamentals.

## What I found

The results....

## Data

The panel has 277,992 rows: 39 extra-EU exporters x 27 EU destinations x 24 HS6 codes x 11 years, with zeros kept as true zeros (74.0% of the panel). I validated it against mirror statistics (EU-reported CIF vs Brazil-reported FOB: +6.1% mean discrepancy, correlation 0.985) and the 2020 pandemic dip (-20.1%) shows up where it should.

Sources:

- Eurostat Comext (dataset DS-045409) for the import values
- ComexStat/MDIC for Brazil's own export records (mirror check)
- World Bank WDI for GDP and population
- CEPII Gravity database V202211 for distance, contiguity, language and RTA

I don't redistribute the raw files (size and licensing), but scripts 01 to 04 re-download everything with the same queries I used. `data/DOWNLOAD_LOG.md` has the exact queries, dates and row counts. Trade statistics get revised over time, so a fresh download can differ a bit from the reported numbers.

