
import pandas as pd

# Load chatbot execution results
df = pd.read_csv("dataset/chain_results.csv")

# Display the first five records
print("\n===== CHATBOT RESULTS =====")
print(df.head())

# Calculate performance
total = len(df)
successful = (df["status"] == "Success").sum()
failed = (df["status"] == "Failed").sum()

success_rate = (successful / total * 100) if total > 0 else 0

print("\n===== PERFORMANCE ANALYSIS =====")
print("Total Questions:", total)
print("Successful Answers:", successful)
print("Failed Answers:", failed)
print("Success Rate:", round(success_rate, 2), "%")

# Analyse each prompt template
if "template_name" in df.columns:
    print("\n===== TEMPLATE-WISE ANALYSIS =====")

    analysis = df.groupby("template_name").agg(
        total_questions=("status", "count"),
        successful=("status", lambda x: (x == "Success").sum()),
        failed=("status", lambda x: (x == "Failed").sum())
    )

    print(analysis)

print("\nAnalysis completed successfully!")
