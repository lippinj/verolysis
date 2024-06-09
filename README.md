# Verolysis

Breaks down the truth of Finnish tax statistics!

## Quick start

```py
import verolysis

df = verolysis.data.ansiotulot(2022, "Y")
f = verolysis.income_brackets.to_density(df)
verolysis.plot.plot(f, xmax=100_000, ymax=100, figsize=(10, 3))
```
