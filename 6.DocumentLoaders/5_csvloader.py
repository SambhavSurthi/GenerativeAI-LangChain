from langchain_community.document_loaders import CSVLoader

loader=CSVLoader(file_path='text.csv')
docs=loader.load()

print(docs)
print()
print()
print(docs[0])
print()
print()
print(docs[2])