from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text='''

# Text Splitters

it is the process of breaking large chunks of text(like articals, PDFs, HTML pages or books) into smaller managable pieces(chunks) that an LLM can handle effectively

over comes model limitations

it helps in

- Embeddings: more accurate embedding
- semantic search: search result point to focused info, not noise
- summarization: prevents hallucination and topic drift

workinf with samller chunkks of text can be more memory efficeint and allow for better parallization of processing task

Types of text splitting 

1. length based splitting
2. text structure based
3. document structure baased
4. semantic meaning based

### 1. Lenghth based text splitting

no grammer, no meaning, not completion og words

### 2. text Structure based

most used: recursaive charector text splitting

### 3. Document-Structured based

used for code, html, markdowns, list..etc

here also we use recursive splitter we just add speraters for chunking. 

### 4. Semantic meaning based





'''

doc_splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0
)

splitted_text=doc_splitter.split_text(text=text)

print(splitted_text)

print(len(splitted_text))