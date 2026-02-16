from unstructured.partition.auto import partition
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import io

def parse_with_unstructured(file_bytes):
    """Local layout-aware parsing."""
    elements = partition(file_bytes=io.BytesIO(file_bytes))
    text = "\n".join([e.text for e in elements])
    return text

def parse_with_azure(file_bytes, endpoint, key):
    """Enterprise OCR using Azure Document Intelligence."""
    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-layout", file=io.BytesIO(file_bytes))
    result = poller.result()
    text = "\n".join([p.content for p in result.pages])
    return text
