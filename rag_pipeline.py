import os
import sys
import time
from dotenv import load_dotenv
import fitz  # PyMuPDF
import voyageai
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Force UTF-8 stdout so extracted PDF text with non-ASCII symbols (e.g. Greek
# letters in equations) doesn't crash printing on Windows' cp1252 console.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 1. Load environment variables using python-dotenv
load_dotenv()

# Global variables storing state
DOCS = {}
fixed_chunks = []
semantic_chunks = []

# Initialize Voyage AI, Groq, and Qdrant in-memory client
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
qdrant = QdrantClient(":memory:")

# ==================================================
# Step 1: Document Ingestion
# ==================================================

def load_pdf_pages(file_path: str, start_page: int = 1, end_page: int = None) -> str:
    """
    Extract text from a specified page range of a PDF file using PyMuPDF (fitz).
    
    Parameters:
        file_path (str): Absolute or relative path to the PDF file.
        start_page (int): 1-based start page number (inclusive, default: 1).
        end_page (int): 1-based end page number (inclusive, default: None for all pages to end).
        
    Returns:
        str: Concatenated text extracted from the specified page range.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    # Convert 1-based page numbers to 0-based indexing for PyMuPDF
    start_idx = max(0, start_page - 1)
    if end_page is None:
        end_idx = total_pages
    else:
        end_idx = min(end_page, total_pages)
        
    page_texts = []
    for page_num in range(start_idx, end_idx):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        block_strings = [b[4].strip() for b in blocks if len(b) >= 5 and b[4].strip()]
        if block_strings:
            page_texts.append("\n\n".join(block_strings))
        
    doc.close()
    return "\n\n".join(page_texts)

def run_ingestion(docs_dir: str = "docs") -> dict:
    """
    Load specified PDF files and page ranges into a dictionary.
    """
    document_specs = [
        {"filename": "nvidia_10k.pdf", "start_page": 35, "end_page": 50},
        {"filename": "attention_paper.pdf", "start_page": 1, "end_page": 15},
        {"filename": "nist_framework.pdf", "start_page": 10, "end_page": 25},
    ]

    extracted_docs = {}
    print("==================================================")
    print("  RAG Pipeline - Step 1: Document Ingestion")
    print("==================================================")

    for spec in document_specs:
        filename = spec["filename"]
        start_page = spec["start_page"]
        end_page = spec["end_page"]
        file_path = os.path.join(docs_dir, filename)

        print(f"\n[*] Starting text extraction for '{filename}' (pages {start_page} to {end_page})...")
        text = load_pdf_pages(file_path, start_page=start_page, end_page=end_page)
        extracted_docs[filename] = text
        print(f"    -> Extracted {len(text):,} characters from '{filename}'.")

    print("\n==================================================")
    print(f"[+] Total number of documents successfully loaded into memory: {len(extracted_docs)}")
    print("==================================================")

    return extracted_docs


# ==================================================
# Step 2: Document Chunking
# ==================================================

def fixed_size_chunking(text: str, source: str, size: int = 150, overlap: int = 30) -> list:
    """
    Splits text into word-count chunks using a sliding window with overlap.
    """
    words = text.split()
    if not words:
        return []

    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + size]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "source": source
        })
        if i + size >= len(words):
            break

    return chunks

def split_oversized_paragraph(paragraph: str, max_words: int) -> list:
    """
    Recursively breaks a single paragraph that exceeds max_words into smaller
    units, trying progressively finer boundaries: single newlines, then
    sentence boundaries, then a hard word-count slice as the last resort.
    """
    if len(paragraph.split()) <= max_words:
        return [paragraph]

    lines = [l.strip() for l in paragraph.split("\n") if l.strip()]
    if len(lines) > 1:
        pieces = []
        for line in lines:
            pieces.extend(split_oversized_paragraph(line, max_words))
        return pieces

    sentences = [s.strip() for s in paragraph.split(". ") if s.strip()]
    if len(sentences) > 1:
        pieces = []
        for sentence in sentences:
            if not sentence.endswith((".", "!", "?")):
                sentence += "."
            pieces.extend(split_oversized_paragraph(sentence, max_words))
        return pieces

    # Last resort: no natural boundary found, hard-slice by word count.
    words = paragraph.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def semantic_chunking(text: str, source: str, max_words: int = 200, overlap_words: int = 40) -> list:
    """
    Recursive-merge chunker: splits text into paragraphs on double newlines,
    then sequentially merges those paragraphs into chunks of up to max_words,
    seeding each new chunk with the last overlap_words words of the previous
    one to preserve context continuity across chunk boundaries. Paragraphs
    that alone exceed max_words are recursively split first (see
    split_oversized_paragraph) so no oversized fragment breaks the merge loop.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    units = []
    for paragraph in paragraphs:
        units.extend(split_oversized_paragraph(paragraph, max_words))

    chunks = []
    current_words = []

    for unit in units:
        unit_words = unit.split()
        if not unit_words:
            continue

        if current_words and len(current_words) + len(unit_words) > max_words:
            chunks.append(" ".join(current_words))
            current_words = current_words[-overlap_words:] if overlap_words > 0 else []

        current_words.extend(unit_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [{"text": chunk_text, "source": source} for chunk_text in chunks]

def run_chunking(docs: dict) -> tuple:
    """
    Processes documents in `docs` using both fixed-size and semantic chunking strategies,
    assigns unique integer IDs (0..N-1 and 0..M-1), and prints statistics and previews.
    """
    f_chunks = []
    s_chunks = []

    for source, text in docs.items():
        f_chunks.extend(fixed_size_chunking(text, source, size=150, overlap=30))
        s_chunks.extend(semantic_chunking(text, source))

    # Assign unique integer IDs
    for idx, chunk in enumerate(f_chunks):
        chunk["id"] = idx

    for idx, chunk in enumerate(s_chunks):
        chunk["id"] = idx

    print("\n==================================================")
    print("  RAG Pipeline - Step 2: Document Chunking")
    print("==================================================")
    print(f"Total fixed-size chunks created: {len(f_chunks)}")
    print(f"Total semantic chunks created:   {len(s_chunks)}")

    print("\n--- Preview of First Fixed-Size Chunk ---")
    if f_chunks:
        first_f = f_chunks[0]
        preview_f = first_f["text"][:100].replace("\n", " ")
        print(f"ID: {first_f['id']} | Source: '{first_f['source']}'")
        print(f"Text Preview: \"{preview_f}...\"")

    print("\n--- Preview of First Semantic Chunk ---")
    if s_chunks:
        first_s = s_chunks[0]
        preview_s = first_s["text"][:100].replace("\n", " ")
        print(f"ID: {first_s['id']} | Source: '{first_s['source']}'")
        print(f"Text Preview: \"{preview_s}...\"")

    print("==================================================")

    return f_chunks, s_chunks


# ==================================================
# Step 3 & 4: Embeddings & Vector Storage
# ==================================================

def create_batches(chunks: list, max_words_per_batch: int = 4000) -> list:
    """
    Groups chunks into batches so that cumulative words per batch stay under max_words_per_batch.
    This guarantees staying within Voyage API token rate limits (10K TPM free tier).
    """
    batches = []
    current_batch = []
    current_words = 0

    for chunk in chunks:
        chunk_words = len(chunk["text"].split())
        if current_batch and (current_words + chunk_words > max_words_per_batch):
            batches.append(current_batch)
            current_batch = [chunk]
            current_words = chunk_words
        else:
            current_batch.append(chunk)
            current_words += chunk_words

    if current_batch:
        batches.append(current_batch)

    return batches

def embed_with_retry(vo_client, batch_texts: list, model_name: str = "voyage-3-lite", input_type: str = "document", max_retries: int = 5) -> list:
    """
    Calls Voyage AI embed API with exponential retry logic on rate limits.
    """
    for attempt in range(max_retries):
        try:
            res = vo_client.embed(batch_texts, model=model_name, input_type=input_type)
            return res.embeddings
        except Exception as e:
            err_msg = str(e)
            if "RateLimitError" in err_msg or "rate" in err_msg.lower() or "429" in err_msg:
                wait_time = 22 * (attempt + 1)
                print(f"    [!] Rate limit encountered. Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Max retries exceeded for Voyage AI embedding request.")

def embed_and_store(chunks: list, collection_name: str, model_name: str = "voyage-3-lite", max_words_per_batch: int = 4000, delay_between_batches: float = 21.0) -> int:
    """
    Extracts text strings from `chunks`, generates embeddings using Voyage AI (`model_name`),
    creates/resets a Qdrant collection, and uploads vector points with payload.
    
    Parameters:
        chunks (list[dict]): List of chunk dictionaries with 'id', 'text', and 'source'.
        collection_name (str): Target Qdrant collection name.
        model_name (str): Voyage AI embedding model (default: 'voyage-3-lite').
        max_words_per_batch (int): Max cumulative words per API call.
        delay_between_batches (float): Pause between API calls to honor 3 RPM free tier.
        
    Returns:
        int: Total number of vectors successfully stored.
    """
    if not chunks:
        print(f"[!] No chunks provided for collection '{collection_name}'. Skipping.")
        return 0

    print(f"\n[*] Generating embeddings for {len(chunks)} chunks in '{collection_name}' using '{model_name}'...")
    batches = create_batches(chunks, max_words_per_batch=max_words_per_batch)
    total_batches = len(batches)

    all_embeddings = []
    for idx, batch in enumerate(batches, 1):
        batch_texts = [c["text"] for c in batch]
        word_count = sum(len(t.split()) for t in batch_texts)
        print(f"    -> Batch {idx}/{total_batches}: Embedding {len(batch_texts)} chunks (~{word_count:,} words)...")

        embeddings = embed_with_retry(vo, batch_texts, model_name=model_name, input_type="document")
        all_embeddings.extend(embeddings)

        if idx < total_batches:
            time.sleep(delay_between_batches)

    vector_size = len(all_embeddings[0])
    print(f"[+] Embeddings generated successfully! Vector dimension: {vector_size}")

    # Recreate or create collection in Qdrant
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    # Format points with unique IDs, vector embeddings, and payload (text + source)
    points = [
        PointStruct(
            id=chunk["id"],
            vector=embedding,
            payload={
                "text": chunk["text"],
                "source": chunk["source"]
            }
        )
        for chunk, embedding in zip(chunks, all_embeddings)
    ]

    print(f"[*] Uploading {len(points)} points to Qdrant collection '{collection_name}'...")
    qdrant.upsert(collection_name=collection_name, points=points)

    collection_info = qdrant.get_collection(collection_name)
    stored_count = collection_info.points_count
    print(f"[+] Successfully stored {stored_count} vectors in collection '{collection_name}'.")

    return stored_count

def run_embeddings_and_storage(f_chunks: list, s_chunks: list):
    """
    Process and store fixed_chunks into 'collection_fixed' and semantic_chunks into 'collection_semantic'.
    """
    print("\n==================================================")
    print("  RAG Pipeline - Step 3 & 4: Embeddings & Storage")
    print("==================================================")

    count_fixed = embed_and_store(f_chunks, "collection_fixed")
    count_semantic = embed_and_store(s_chunks, "collection_semantic")

    print("\n==================================================")
    print("  Vector Storage Verification Summary")
    print("==================================================")
    print(f"[+] Total vectors stored in 'collection_fixed':    {count_fixed}")
    print(f"[+] Total vectors stored in 'collection_semantic': {count_semantic}")
    print("==================================================")


# ==================================================
# Step 5: Retrieval & Generation
# ==================================================

def retrieve(query: str, collection_name: str, top_k: int = 5) -> list:
    """
    Embeds `query` (input_type='query') and retrieves the top_k nearest chunks
    from the given Qdrant collection.

    Returns:
        list[dict]: Each dict has 'text', 'source', and 'score'.
    """
    query_embedding = embed_with_retry(vo, [query], input_type="query")[0]

    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
    ).points

    return [
        {
            "text": point.payload["text"],
            "source": point.payload["source"],
            "score": point.score,
        }
        for point in results
    ]

def build_prompt(query: str, context_chunks: list) -> str:
    """
    Assembles a grounded RAG prompt from retrieved chunks, numbered so the
    model can cite which source each fact came from.
    """
    context_blocks = [
        f"[{i}] (source: {chunk['source']})\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks, 1)
    ]
    context_text = "\n\n".join(context_blocks)

    return (
        "Answer the question using ONLY the context below. "
        "Cite sources using the [n] markers. If the context does not contain "
        "the answer, say so explicitly instead of guessing.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

def generate_answer(query: str, context_chunks: list, model_name: str = "llama-3.3-70b-versatile") -> str:
    """
    Sends the RAG prompt to Groq's chat completion API and returns the answer text.
    """
    prompt = build_prompt(query, context_chunks)

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a precise, grounded research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

def run_rag_query(query: str, top_k: int = 5) -> dict:
    """
    Runs the full retrieval + generation flow against both the fixed-size and
    semantic collections, then prints a side-by-side comparison.

    Returns:
        dict: {'fixed': {...}, 'semantic': {...}} with retrieved chunks and answers.
    """
    print("\n==================================================")
    print("  RAG Pipeline - Step 5: Retrieval & Generation")
    print("==================================================")
    print(f"Query: \"{query}\"")

    results = {}
    for label, collection_name in [("fixed", "collection_fixed"), ("semantic", "collection_semantic")]:
        print(f"\n--- Retrieval: {label.upper()} chunking ({collection_name}) ---")
        chunks = retrieve(query, collection_name, top_k=top_k)
        for i, c in enumerate(chunks, 1):
            preview = c["text"][:100].replace("\n", " ")
            print(f"  [{i}] score={c['score']:.4f} source={c['source']} | \"{preview}...\"")

        print(f"\n--- Generation: {label.upper()} chunking ---")
        answer = generate_answer(query, chunks)
        print(answer)

        results[label] = {"chunks": chunks, "answer": answer}

    print("\n==================================================")
    return results


if __name__ == "__main__":
    DOCS = run_ingestion()
    fixed_chunks, semantic_chunks = run_chunking(DOCS)
    run_embeddings_and_storage(fixed_chunks, semantic_chunks)

    sample_query = "What are the main components of the Transformer architecture?"
    run_rag_query(sample_query, top_k=5)