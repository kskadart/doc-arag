#!/usr/bin/env python3
"""Simple test client for embedding service connectivity."""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.docarag.clients.embedding import EmbeddingGRPCClient
from src.docarag.config import settings


async def test_async_client():
    """Test async gRPC client."""
    print("Testing async gRPC embedding client...")

    client = EmbeddingGRPCClient(use_async=True)

    try:
        # Test single text embedding
        print("Testing single text embedding...")
        text = "This is a test sentence for embedding generation."
        embedding = await client.embed_text_async(text)
        print(f"✓ Single embedding generated: {len(embedding)} dimensions")
        print(f"  First 5 values: {embedding[:5]}")

        # Test batch embedding
        print("Testing batch embedding...")
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]
        embeddings = await client.embed_batch_async(texts)
        print(f"✓ Batch embeddings generated: {len(embeddings)} embeddings")
        print(f"  Each embedding has {len(embeddings[0])} dimensions")

        # Test dimension query
        print("Testing dimension query...")
        dimension = await client.get_embedding_dimension_async()
        print(f"✓ Embedding dimension: {dimension}")

    except Exception as e:
        print(f"✗ Error during async test: {e}")
        return False
    finally:
        await client.close_async()

    return True


def test_sync_client():
    """Test sync gRPC client."""
    print("\nTesting sync gRPC embedding client...")

    client = EmbeddingGRPCClient(use_async=False)

    try:
        # Test single text embedding
        print("Testing single text embedding...")
        text = "This is a test sentence for embedding generation."
        embedding = client.embed_text(text)
        print(f"✓ Single embedding generated: {len(embedding)} dimensions")
        print(f"  First 5 values: {embedding[:5]}")

        # Test batch embedding
        print("\nTesting batch embedding...")
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]
        embeddings = client.embed_batch(texts)
        print(f"✓ Batch embeddings generated: {len(embeddings)} embeddings")
        print(f"  Each embedding has {len(embeddings[0])} dimensions")

        # Test dimension query
        print("Testing dimension query...")
        dimension = client.get_embedding_dimension()
        print(f"✓ Embedding dimension: {dimension}")

    except Exception as e:
        print(f"✗ Error during sync test: {e}")
        return False
    finally:
        client.close()

    return True


async def main():
    """Main test function."""
    print("=" * 60)
    print("Embedding Service Connectivity Test")
    print("=" * 60)
    print(f"Service URL: {settings.embedding_service_url}")
    print(f"Timeout: {settings.embedding_service_timeout}s")
    print(f"Default async mode: {settings.embedding_use_async}")
    print("=" * 60)

    # Test async client
    async_success = await test_async_client()

    # Test sync client
    sync_success = test_sync_client()

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Async client: {'✓ PASS' if async_success else '✗ FAIL'}")
    print(f"Sync client:  {'✓ PASS' if sync_success else '✗ FAIL'}")
    print("=" * 60)

    if not (async_success and sync_success):
        print("\n⚠️  Note: Tests are using placeholder implementations.")
        print(
            "   Copy the actual *_pb2*.py files to src/docarag/proto/ to enable real gRPC calls."
        )
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
