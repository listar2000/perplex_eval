import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=[" ", "\n\n", "\n"]
)

chunked_rows = []

# Use the 'id' column from unique_df for text_id
for _, row in unique_df.iterrows():
    text = row['text']
    text_id = row['id']   # Use the id from the original df
    chunks = splitter.split_text(text)
    for chunk_id, chunk in enumerate(chunks):
        chunked_rows.append({
            "text_id": text_id,
            "chunk_id": chunk_id,
            "chunk_text": chunk
        })

chunked_df = pd.DataFrame(chunked_rows)

# Save as parquet
chunked_df.to_parquet("chunked_texts_df.parquet", index=False)

print(chunked_df.head())
