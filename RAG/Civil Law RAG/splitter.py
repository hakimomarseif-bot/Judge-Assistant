"""
splitter.py

Egyptian Civil Law hierarchical text splitter.

Purpose:
---------
Transforms the raw civil law text file into structured LangChain
Document objects with hierarchical metadata.

It detects and separates:
- Book (الكتاب)
- Part (الباب)
- Chapter (الفصل)
- Article (المادة)

Each generated Document includes metadata:
{
    "type": "article | chapter | part | book",
    "title": "...",
    "index": int (for articles),
    "book": "...",
    "part": "...",
    "chapter": "..."
}

Why this exists:
----------------
Accurate legal retrieval requires semantic chunks aligned with
real legal structure. Articles must be independently retrievable.

This module is purely structural.
It does NOT perform:
- Embeddings
- Vector storage
- Retrieval
- LLM processing

Design Principle:
-----------------
Separation of concerns.
Text parsing logic must be isolated from vector indexing and AI layers.
"""
import re
from langchain_core.documents import Document 


def split_egyptian_civil_law(text: str):
    """
    Splits Egyptian Civil Law text into Documents with types:
    - book, part, chapter, article
    - Avoids repeating books if they appear multiple times
    - Attaches hierarchy info in metadata: book, part, chapter
    - Supports preliminary باب (الباب التمهيدي)
    """
    # Heading patterns
    book_pat    = r"^الكتاب[^\n]*"
    part_pat    = r"^(الباب\s+التمهيدي|الباب[^\n]*)"
    chapter_pat = r"^الفصل[^\n]*"
    article_pat = r"^المادة\s*([0-9]+)"

    combined = rf"(?m)(?=({book_pat}|{part_pat}|{chapter_pat}|{article_pat}))"
    raw_sections = re.split(combined, text)
    sections = [s.strip() for s in raw_sections if s and s.strip()]

    docs = []
    last_heading_normalized = None
    seen_books = set()  # track unique books

    # Current hierarchy
    current_book = None
    current_part = None
    current_chapter = None

    re_book = re.compile(book_pat, flags=re.MULTILINE)
    re_part = re.compile(part_pat, flags=re.MULTILINE)
    re_chap = re.compile(chapter_pat, flags=re.MULTILINE)
    re_article = re.compile(article_pat, flags=re.MULTILINE)

    for sec in sections:
        m_article = re_article.match(sec)
        m_book = re_book.match(sec)
        m_part = re_part.match(sec)
        m_chap = re_chap.match(sec)

        # Handle articles
        if m_article:
            heading_raw = m_article.group(0).strip()
            heading_norm = heading_raw
            if heading_norm == last_heading_normalized:
                if docs:
                    docs[-1] = Document(
                        page_content=docs[-1].page_content + "\n" + sec[m_article.end():].strip(),
                        metadata=docs[-1].metadata
                    )
                continue
            last_heading_normalized = heading_norm

            body = sec[m_article.end():].strip()
            index = int(m_article.group(1))

            meta = {
                "type": "article",
                "title": heading_raw,
                "index": index,
                "book": current_book,
                "part": current_part,
                "chapter": current_chapter
            }
            docs.append(Document(page_content=f"{heading_raw}\n{body}", metadata=meta))
            continue

        # Handle books, parts, chapters
        if m_book or m_part or m_chap:
            m = m_book or m_part or m_chap
            heading_raw = m.group(0).strip()
            heading_norm = heading_raw
            if heading_norm == last_heading_normalized:
                if docs:
                    docs[-1] = Document(
                        page_content=docs[-1].page_content + "\n" + sec[m.end():].strip(),
                        metadata=docs[-1].metadata
                    )
                continue
            last_heading_normalized = heading_norm

            body = sec[m.end():].strip()

            if m is m_book:
                t = "book"
                # skip repeated books but keep hierarchy
                if heading_norm in seen_books:
                    current_book = heading_norm
                    current_part = None
                    current_chapter = None
                    continue
                seen_books.add(heading_norm)
                current_book = heading_raw
                current_part = None
                current_chapter = None
            elif m is m_part:
                t = "part"
                current_part = heading_raw
                current_chapter = None
            else:
                t = "chapter"
                current_chapter = heading_raw

            meta = {
                "type": t,
                "title": heading_raw,
                "book": current_book,
                "part": current_part,
                "chapter": current_chapter
            }
            docs.append(Document(page_content=f"{heading_raw}\n{body}", metadata=meta))
            continue

        # Any other text → preface
        if docs:
            docs[-1] = Document(
                page_content=docs[-1].page_content + "\n" + sec.strip(),
                metadata=docs[-1].metadata
            )
        else:
            docs.append(Document(page_content=sec.strip(), metadata={"type":"preface", "title":"المقدمة"}))

    # # Post-process: add adjacent article overlap for cross-article reasoning.
    # # Each article chunk gets a preview of its neighbors so the embedding
    # # captures context that spans article boundaries.
    # docs = _add_article_overlap(docs)

    return docs

# Note: The adjacent article overlap logic is currently disabled to keep article chunks clean and focused. It can be re-enabled if retrieval quality suffers due to lack of cross-article context.
# def _add_article_overlap(docs: list, context_chars: int = 200) -> list:
#     """Enrich each article Document with a short preview of its neighboring
#     articles so that retrieval can capture cross-article context.

#     Parameters
#     ----------
#     docs : list[Document]
#         The list of Documents produced by ``split_egyptian_civil_law``.
#     context_chars : int
#         Maximum characters to include from each neighbor.

#     Returns
#     -------
#     list[Document]
#         The same list, with article page_content augmented in-place.
#     """
#     article_indices = [
#         i for i, d in enumerate(docs) if d.metadata.get("type") == "article"
#     ]

#     for pos, idx in enumerate(article_indices):
#         prev_snippet = ""
#         next_snippet = ""

#         # Previous article
#         if pos > 0:
#             prev_idx = article_indices[pos - 1]
#             prev_text = docs[prev_idx].page_content
#             prev_snippet = prev_text[:context_chars]

#         # Next article
#         if pos < len(article_indices) - 1:
#             next_idx = article_indices[pos + 1]
#             next_text = docs[next_idx].page_content
#             next_snippet = next_text[:context_chars]

#         overlap_parts = []
#         if prev_snippet:
#             overlap_parts.append(f"[سياق المادة السابقة]: {prev_snippet}...")
#         if next_snippet:
#             overlap_parts.append(f"[سياق المادة التالية]: {next_snippet}...")

#         if overlap_parts:
#             overlap_text = "\n".join(overlap_parts)
#             docs[idx] = Document(
#                 page_content=f"{docs[idx].page_content}\n\n{overlap_text}",
#                 metadata=docs[idx].metadata,
#             )

#     return docs