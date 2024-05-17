# Verolysis

Breaks down the truth of Finnish tax statistics!

## Getting data

Tax statistics are published online in
[finnish](https://vero2.stat.fi/PXWeb/pxweb/fi/Vero/),
[swedish](https://vero2.stat.fi/PXWeb/pxweb/se/Vero/) and
[english](https://vero2.stat.fi/PXWeb/pxweb/en/Vero/).
Generate a result table on the website,
then download the data in JSON format.

## Usage

Load the data from one or more JSON files:


```py
import verolysis

# Load from your JSON file
table = verolysis.load("hvt_tulot_101.json")

# The table contains a Pandas dataframe
table.df

# You can add more data from other tables, as long as the columns match
table.load("hvt_verot_101.json")
```

There are some general convenience features to be aware of:

```py
# They key columns are coded.
# For some codes, you can find labels in verolysis.codes:
assert verolysis.codes.fi.Erä["HVT_TULOT_60"] == "HVT_TULOT_60"
assert verolysis.codes.fi.Tulonsaajaryhmä["2"] == "Eläkeläinen"
assert verolysis.codes.fi.Tuloluokka["5"] == "20 000 - 25 999"

# Select only certain rows, based on keys
df = table.where(
    Verovuosi = 2022,
    Tulonsaajaryhmä = 2,
    Erä = "HVT_TULOT_50",
)
```