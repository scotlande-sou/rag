import os
from datetime import timedelta
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Union
import re
import openai
import argparse
from dotenv import load_dotenv
load_dotenv()

# Import VTTProcessor from the previous code
class VTTProcessor:
    def __init__(self, time_window: timedelta = timedelta(minutes=5), max_chunk_length: int = 1000):
        self.time_window = time_window
        self.max_chunk_length = max_chunk_length

    @staticmethod
    def parse_vtt_timestamps(timestamp: str) -> timedelta:
        try:
        # Split timestamp into hours, minutes, and seconds
            time_parts = timestamp.split(":")
            if len(time_parts) != 3:
                raise ValueError(f"Invalid timestamp format: {timestamp}")
        
            hours, minutes = map(int, time_parts[:2])
            # Handle seconds and milliseconds
            seconds_parts = time_parts[2].split(".")
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0

            return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid timestamp format: {timestamp}") from e
    @staticmethod
    def parse_vtt_content(vtt_lines: List[str]) -> List[Dict[str, Union[str, timedelta]]]:
        subtitles = []
        i = 0
        while i < len(vtt_lines):
            if "-->" in vtt_lines[i]:
                try:
                    start, end = map(str.strip, vtt_lines[i].split("-->"))
                    start_time = VTTProcessor.parse_vtt_timestamps(start)
                    end_time = VTTProcessor.parse_vtt_timestamps(end)
                    text = []
                    speaker = None
                    i += 1
                    while i < len(vtt_lines) and vtt_lines[i].strip():
                        line = vtt_lines[i].strip()
                        if ":" in line and not text:
                            speaker, line = map(str.strip, line.split(":", 1))
                        text.append(line)
                        i += 1
                    subtitles.append({
                        "start": start_time,
                        "end": end_time,
                        "text": " ".join(text),
                        "speaker": speaker,
                })
                except ValueError as e:
                    print(f"Skipping invalid VTT line at {i}: {e}")
            i += 1
        return subtitles

    def chunk_subtitles(self, subtitles: List[Dict[str, Union[str, timedelta]]]) -> List[Dict[str, Union[str, timedelta, int]]]:
        chunks = []
        chunk = {"id": 1, "start": None, "end": None, "text": [], "speakers": set()}
        for subtitle in subtitles:
            if not chunk["start"]:
                chunk["start"] = subtitle["start"]
            if not chunk["end"] or subtitle["end"] <= chunk["start"] + self.time_window:
                chunk["text"].append(subtitle["text"])
                chunk["speakers"].add(subtitle["speaker"])
                chunk["end"] = subtitle["end"]
            else:
                full_text = " ".join(chunk["text"])
                while len(full_text) > self.max_chunk_length:
                    split_idx = full_text.rfind(" ", 0, self.max_chunk_length)
                    chunks.append({
                        "id": chunk["id"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "text": " ".join(chunk["text"]),
                        "speakers": list(chunk["speakers"]),
                    })
                    full_text = full_text[split_idx:]
                    chunk["id"] += 1

            # Add the remainder of the chunk
                chunks.append({
                    "id": chunk["id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": full_text,
                    "speakers": list(chunk["speakers"]),
                })

                chunk = {
                    "id": chunk["id"] + 1,
                    "start": subtitle["start"],
                    "end": subtitle["end"],
                    "text": [subtitle["text"]],
                    "speakers": {subtitle["speaker"]},
                }
        if chunk["text"]:
            chunks.append({
                "id": chunk["id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "text": " ".join(chunk["text"]),
                "speakers": list(chunk["speakers"]),
            })
        return chunks

    def process_vtt(self, vtt_content: str) -> List[Dict[str, Union[str, timedelta, int]]]:
        vtt_lines = vtt_content.splitlines()
        subtitles = self.parse_vtt_content(vtt_lines)
        return self.chunk_subtitles(subtitles)


# Step 1: Initialize ChromaDB and the embedding model
client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_or_create_collection("transcripts")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Step 2: Process and store VTT file
def process_and_store_vtt(vtt_path: str):
    # Read the VTT file
    with open(vtt_path, "r", encoding="utf-8") as file:
        vtt_content = file.read()
    
    # Chunk the content
    processor = VTTProcessor(time_window=timedelta(minutes=5))
    chunks = processor.process_vtt(vtt_content)

    batch_size = 32  # Customize batch size for efficiency
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        text_chunks = [chunk["text"] for chunk in batch_chunks]
        embeddings = embedding_model.encode(text_chunks).tolist()
        metadatas = [
            {"id": chunk["id"], "start": str(chunk["start"]), "end": str(chunk["end"]), "speakers": ", ".join(chunk["speakers"])}
            for chunk in batch_chunks
        ]
        ids = [str(chunk["id"]) for chunk in batch_chunks]
        collection.add(documents=text_chunks, metadatas=metadatas, ids=ids)


def query_llm(prompt: str) -> str:
    """
    Process a query through OpenAI's ChatCompletion endpoint.
    :param prompt: The prompt to send to the model.
    :return: The response from the chat model.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Or "gpt-3.5-turbo" for lower cost
            messages=[
                {"role": "system", "content": "You are an AI assistant helping with meeting transcription insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def process_user_query(query: str, top_n: int = 5) -> str:
    """
    Process a user query using embeddings and an LLM.
    :param query: User's query string.
    :param top_n: Number of top results to retrieve from ChromaDB.
    :return: Generated response from the LLM.
    """
    # Step 1: Encode the query into embeddings
    query_embedding = embedding_model.encode([query]).tolist()[0]

    # Step 2: Perform similarity search in ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=top_n)

    # Step 3: Extract relevant context
    retrieved_chunks = results.get("documents", [[]])[0]  # List of text chunks
    retrieved_metadata = results.get("metadatas", [[]])[0]  # Corresponding metadata

    # Step 4: Format the context for the LLM
    context = "\n\n".join(
        [f"Chunk {i+1} (Start: {meta['start']}, End: {meta['end']}): {text}"
         for i, (text, meta) in enumerate(zip(retrieved_chunks, retrieved_metadata))]
    )
    prompt = (
        f"You are an AI assistant. Use the following transcript chunks to answer the question:\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer as concisely as possible:"
    )

    # Step 5: Query the LLM and return the result
    return query_llm(prompt)

if __name__ == "__main__":
    vtt_file = "GMT20241028-223931_Recording.transcript.vtt"  # Path to your VTT file

    parser = argparse.ArgumentParser(description="Search transcripts in ChromaDB.")
    parser.add_argument("--vtt", type=str, help="Path to the VTT file to process.")
    parser.add_argument("--query", type=str, help="Search query for the processed transcripts.")
    parser.add_argument("--reset", action="store_true", help="Clear the ChromaDB database.")
    args = parser.parse_args()
    

    if args.reset:
        client.delete_collection("transcripts")  # Deletes the collection entirely
        collection = client.get_or_create_collection("transcripts")  # Recreates the empty collection
        print("Database has been cleared.")
    elif args.vtt:
        process_and_store_vtt(args.vtt)
    elif args.query:
        response = process_user_query(args.query, top_n=5)
        print(f"Response:\n{response}")
    else:
        parser.print_help()

    # Test retrieval
    #query_text = "What is the main topic of the session?"
    #response = process_user_query(query_text, top_n=5)
    #print("LLM Response:\n", response)
