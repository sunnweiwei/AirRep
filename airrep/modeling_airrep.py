from typing import List, Optional, Union, Dict, Any
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, PreTrainedModel, AutoTokenizer
import os
import numpy as np
from tqdm import tqdm


def mean_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply mean pooling to hidden states."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class AirRepConfig(BertConfig):
    """Configuration class for AirRep model."""

    model_type = "airrep"

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)


class AirRepModel(PreTrainedModel):
    """
    AirRep model with BERT encoder and projection layer.

    This is a standalone model, not a wrapper.
    """

    config_class = AirRepConfig
    base_model_prefix = "airrep"

    def __init__(self, config: AirRepConfig):
        super().__init__(config)
        self.config = config

        # BERT encoder
        self.bert = BertModel(config, add_pooling_layer=False)

        # Projection layer
        self.projector = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            dtype=torch.bfloat16
        )

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs

        Returns:
            Pooled and projected embeddings (batch_size, hidden_size)
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pooled = mean_pooling(last_hidden_state, attention_mask)

        # Project
        projected = self.projector(pooled)

        return projected


class AirRep:
    """Attribution-friendly embedding model with influence computation."""

    def __init__(
        self,
        model: AirRepModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model = self.model.to(device).eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Load a pretrained AirRep model.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            device: Device to load model on
            **kwargs: Additional arguments

        Returns:
            AirRep instance
        """
        # Load config
        config = AirRepConfig.from_pretrained(model_name_or_path)

        # Load model
        model = AirRepModel.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16
        )

        # Load tokenizer (use gte-small tokenizer by default)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")

        return cls(model, tokenizer, device)

    def _format_item(self, item: Dict[str, Any]) -> str:
        """Format data item into text.
        Supports  formats:
        - {'input': ..., 'output': ...}
        """

        # Step 5: Format with prefix
        prompt = 'Question: ' + " ".join(item['input'].split()[:256]) + '\nAnswer:'
        return prompt + ' ' + item['output']

    def encode(
        self,
        texts: Union[str, List[str], List[Dict[str, Any]]],
        batch_size: int = 128,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text, list of texts, or list of data dicts to encode
                   Supports dicts with {'input': ..., 'output': ...} format
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert output to numpy
            normalize_embeddings: Whether to L2 normalize embeddings

        Returns:
            Embeddings as numpy array or torch tensor
        """
        if isinstance(texts, str):
            texts = [texts]

        # Convert data dicts to texts if needed
        if texts and isinstance(texts[0], dict):
            texts = [self._format_item(item) for item in texts]

        all_embeds = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")

        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeds = self.model(**inputs)
                if normalize_embeddings:
                    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

            all_embeds.append(embeds.cpu())

        result = torch.cat(all_embeds, dim=0)

        if convert_to_numpy:
            result = result.float().numpy()

        return result

    def similarity(
        self,
        query_embeddings: Union[np.ndarray, torch.Tensor],
        corpus_embeddings: Union[np.ndarray, torch.Tensor],
        softmax: bool = True,
        chunk_size: int = 1000,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute similarity scores between query and corpus embeddings.

        Args:
            query_embeddings: Query embeddings (N, D)
            corpus_embeddings: Corpus embeddings (M, D)
            softmax: Whether to apply softmax attention weighting
            chunk_size: Chunk size for corpus to avoid OOM

        Returns:
            Similarity scores (N,) - one score per query
        """
        # Convert to tensors if needed
        is_numpy_input = isinstance(query_embeddings, np.ndarray)
        if is_numpy_input:
            query_embeddings = torch.from_numpy(query_embeddings).float()
        if isinstance(corpus_embeddings, np.ndarray):
            corpus_embeddings = torch.from_numpy(corpus_embeddings).float()

        # Compute similarity matrix in chunks to avoid OOM (N, M)
        num_corpus = corpus_embeddings.shape[0]
        all_scores = []

        for i in range(0, num_corpus, chunk_size):
            end_i = min(i + chunk_size, num_corpus)
            chunk_scores = query_embeddings @ corpus_embeddings[i:end_i].T
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=-1)  # (N, M)

        if softmax:
            # Apply softmax attention and weighted sum for each query
            attn = torch.softmax(scores, dim=-1)
            weighted_scores = (attn * scores).sum(dim=-1)
        else:
            # Simple sum
            weighted_scores = scores.sum(dim=-1)

        # Convert to numpy if input was numpy
        if is_numpy_input:
            return weighted_scores.numpy()
        else:
            return weighted_scores

    def save_pretrained(self, save_directory: str):
        """
        Save model to directory.

        Args:
            save_directory: Directory to save directory
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save model and config using HuggingFace method
        self.model.save_pretrained(save_directory)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload AirRep model",
        private: bool = False,
    ):
        """
        Upload model to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub
            commit_message: Commit message
            private: Whether to make repo private
        """
        # Push model
        self.model.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private
        )

        # Push tokenizer
        self.tokenizer.push_to_hub(
            repo_id,
            commit_message=commit_message,
            private=private
        )