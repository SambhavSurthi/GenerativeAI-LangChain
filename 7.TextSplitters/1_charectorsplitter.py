from langchain_text_splitters import CharacterTextSplitter,TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

doc_loader=PyPDFLoader('resume.pdf')
docs=doc_loader.load()


text='''

The Goldfinch (Dutch: Het puttertje) is an oil painting by Carel Fabritius, a Dutch Golden Age artist, of a life-sized chained European goldfinch. Signed and dated 1654, it is in the collection of the Mauritshuis in the Hague, Netherlands. The work is a trompe-l'œil oil on panel measuring 33.5 by 22.8 centimetres (13.2 in × 9.0 in) that was once part of a larger structure, perhaps a window jamb or a protective cover. It is possible that the painting was in its creator's workshop in Delft at the time of the gunpowder explosion that killed him and destroyed much of the city.

A common and colourful bird with a pleasing song, the goldfinch was a popular pet, and could be taught simple tricks including lifting a thimble-sized bucket of water. It was reputedly a bringer of good health, and was used in Italian Renaissance painting as a symbol of Christian redemption and the Passion of Jesus.

The Goldfinch is unusual for the Dutch Golden Age painting period in the simplicity of its composition and use of illusionary techniques. Following the death of its creator, it was lost for more than two centuries before its rediscovery in Brussels.

An eponymous novel by the American author Donna Tartt won the 2014 Pulitzer Prize for fiction and led to a 2019 film. The painting was featured in a Dutch Golden Age world tour in 2012–2014, and was the centrepiece of a 2026 bird art exhibition at the Mauritshuis.

'''

splitter=CharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=60
)

splitted_text=splitter.split_text(text=text)

print(splitted_text)

print(len(splitted_text))


token_splitter = TokenTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=20,
    chunk_overlap=2
)

splitted_text=token_splitter.split_text(text=text)
print(splitted_text)

print(len(splitted_text))


doc_splitter=TokenTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',
    chunk_size=20,
    chunk_overlap=2
)

doc=doc_splitter.split_documents(docs)

print(doc)
print(doc[0])
print(len(doc))