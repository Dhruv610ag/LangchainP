from crewai_tools.tools.youtube_channel_search_tool import YoutubeChannelSearchTool
from embedchain.embedder.huggingface import HuggingFaceEmbedder

# Use HuggingFace embeddings instead of OpenAI
custom_embedder = HuggingFaceEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

yt_tool = YoutubeChannelSearchTool(
    youtube_channel_handle='@krishnaik06',
    embedding_model=custom_embedder
)
