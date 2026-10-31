from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\njnjp\Downloads\mcp-explained-notes-compressed.pdf")

docs = loader.load()

print(docs[0])