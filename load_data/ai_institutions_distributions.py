from utility.universal_helpers import get_verified_ai_majors
import pandas as pd

universities_programs = get_verified_ai_majors()

unis = [x[0] for x in universities_programs]

print(f"# universities: {len(set(unis))}")
print(f"# programs: {len(universities_programs)}")

df = pd.read_csv("ai_universities_attributes.csv")

# Normalize whitespace and capitalization so "Public " and "public" count together
for col in df.columns.drop("university"):
    df[col] = df[col].astype("string").str.strip().str.lower().replace("", pd.NA)

    counts = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    counts[col] = counts[col].fillna("(blank)")

    print(f"\n{col}")
    print(counts.to_string(index=False))
    print(f"total: {counts['count'].sum()}")

