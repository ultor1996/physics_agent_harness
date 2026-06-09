from smolagents import tool


@tool
def analyze_dataframe(csv_path: str, question: str) -> str:
    """
    Loads a CSV file and answers a question about it using pandas.

    Args:
        csv_path: Path to the CSV file to analyze.
        question: The analysis question to answer about the data.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    summary = f"""
Shape: {df.shape}
Columns: {list(df.columns)}
Dtypes:\n{df.dtypes.to_string()}
Head:\n{df.head().to_string()}
Describe:\n{df.describe().to_string()}
    """

    # Return the summary — the agent's LLM will interpret it to answer `question`
    return f"Question: {question}\n\nData summary:\n{summary}"


# Add more custom tools below as your experiments grow.
# Use @tool for simple stateless tools.
# Subclass Tool for anything needing heavy init (models, DB connections).