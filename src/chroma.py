from chromadb import HttpClient
from embedding_util import CustomEmbeddingFunction

client = HttpClient(host="localhost", port=7100)
print('HEARTBEAT:', client.heartbeat())

collection = client.get_or_create_collection(
    name="test", embedding_function=CustomEmbeddingFunction())

print('COLLECTION:', collection)

documents = [
    "A group of vibrant parrots chatter loudly, sharing stories of their tropical adventures.",
    "The mathematician found solace in numbers, deciphering the hidden patterns of the universe.",
    "The robot, with its intricate circuitry and precise movements, assembles the devices swiftly.",
    "The chef, with a sprinkle of spices and a dash of love, creates culinary masterpieces.",
    "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
    "The detective, with keen observation and logical reasoning, unravels the intricate web of clues.",
    "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
    "In the dense forest, the howl of a lone wolf echoes, blending with the symphony of the night.",
    "The dancer, with graceful moves and expressive gestures, tells a story without uttering a word.",
    "In the quantum realm, particles flicker in and out of existence, dancing to the tunes of probability."
]

# Every document needs an id for Chroma
document_ids = list(map(lambda tup: f"id{tup[0]}", enumerate(documents)))

collection.add(documents=documents, ids=document_ids)

# Define the query text
query = "sunset over the sea"

# Perform the query
result = collection.query(query_texts=[query], n_results=5, include=["documents", 'distances'])

# Unpack the result
ids = result['ids'][0]
documents = result['documents'][0]
distances = result['distances'][0]

for id_, document, distance in zip(ids, documents, distances):
    print(f"ID: {id_}, Document: {document}, Similarity: {1 - distance}")
