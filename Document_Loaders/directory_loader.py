from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

# print(len(docs))
# print(docs[1].metadata)

for doc in docs:
    print(doc.page_content)