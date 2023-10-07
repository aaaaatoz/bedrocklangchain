from langchain.embeddings import BedrockEmbeddings
from scipy import spatial


def cosine_similarity(dataSetI, dataSetII):
    return 1 - spatial.distance.cosine(dataSetI, dataSetII)


embeddings = BedrockEmbeddings()

ds1 = embeddings.embed_query("tiger")
ds2 = embeddings.embed_query("whale")

print(cosine_similarity(ds1, ds2))

