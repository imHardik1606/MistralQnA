# tests/conftest.py
import pytest
import tempfile
import os


@pytest.fixture
def sample_long_text():
    """Provide a long sample text for testing"""
    return "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100


@pytest.fixture
def sample_chunks():
    """Provide sample chunks for testing"""
    return [
        "First chunk of text about artificial intelligence.",
        "Second chunk discussing machine learning algorithms.",
        "Third chunk covering natural language processing.",
        "Fourth chunk about computer vision applications."
    ]


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing"""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [0.2, 0.3, 0.4, 0.5, 0.6]
    ]


@pytest.fixture
def mock_mistral_client():
    """Provide a mocked Mistral client"""
    from unittest.mock import Mock
    
    mock_client = Mock()
    mock_client.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_client.chat.return_value = "Mocked AI response"
    
    return mock_client