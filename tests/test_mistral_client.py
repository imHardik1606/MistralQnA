# tests/test_mistral_client.py
import pytest
from unittest.mock import Mock, patch
from app.mistral_client import MistralClient


class TestMistralClient:
    
    def test_client_initialization(self):
        """Test that client initializes properly"""
        client = MistralClient("test_api_key_123")
        assert client.client is not None
    
    @patch('app.mistral_client.Mistral')
    def test_embed_method(self, mock_mistral_class):
        """Test embedding generation with mocked API"""
        # Setup mock
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        mock_embedding_response = Mock()
        mock_embedding_response.data = [mock_embedding]
        
        mock_client_instance = Mock()
        mock_client_instance.embeddings = Mock()
        mock_client_instance.embeddings.create = Mock(return_value=mock_embedding_response)
        
        mock_mistral_class.return_value = mock_client_instance
        
        # Test
        client = MistralClient("test_key")
        client.client = mock_client_instance  # Replace with mock
        
        texts = ["Hello world", "Test document"]
        embeddings = client.embed(texts)
        
        # Verify
        assert len(embeddings) == 1  # Returns one embedding per call
        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Verify API was called correctly
        mock_client_instance.embeddings.create.assert_called_once()
    
    @patch('app.mistral_client.Mistral')
    def test_chat_method(self, mock_mistral_class):
        """Test chat completion with mocked API"""
        # Setup mock response
        mock_message = Mock()
        mock_message.content = "This is a test response from the AI."
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_chat_response = Mock()
        mock_chat_response.choices = [mock_choice]
        
        # Setup mock client
        mock_client_instance = Mock()
        mock_client_instance.chat = Mock()
        mock_client_instance.chat.complete = Mock(return_value=mock_chat_response)
        
        mock_mistral_class.return_value = mock_client_instance
        
        # Test
        client = MistralClient("test_key")
        client.client = mock_client_instance
        
        prompt = "What is artificial intelligence?"
        response = client.chat(prompt)
        
        # Verify
        assert response == "This is a test response from the AI."
        mock_client_instance.chat.complete.assert_called_once_with(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )


def test_error_handling():
    """Test that errors are raised properly"""
    with patch('app.mistral_client.Mistral') as mock_mistral:
        mock_instance = Mock()
        mock_instance.embeddings = Mock()
        mock_instance.embeddings.create = Mock(side_effect=Exception("API Error"))
        mock_mistral.return_value = mock_instance
        
        client = MistralClient("test_key")
        client.client = mock_instance
        
        with pytest.raises(Exception):
            client.embed(["test"])