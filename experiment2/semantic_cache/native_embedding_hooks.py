"""
Native embedding hooks that extract embeddings from VLM models themselves.
Uses the model's own encoder/decoder representations instead of external encoders.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
from PIL import Image


class NativeEncoderEmbeddingHook:
    """
    Extracts embeddings from the VLM's encoder (vision/multimodal encoder).
    Works with both Qwen3-VL and InternVL models.
    Caches encoder outputs to avoid recomputation during generation.
    """
    
    def __init__(self, layer_name: str = "native_encoder", pooling: str = "mean"):
        """
        Args:
            layer_name: Name for this embedding layer in cache
            pooling: How to pool the encoder outputs ("mean", "first", "last")
        """
        self.layer_name = layer_name
        self.pooling = pooling
        self._warned = False
        # Cache encoder outputs to reuse during generation
        self._cached_encoder_outputs: dict[str, Any] = {}
    
    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        """Extract encoder embedding from the model and cache intermediate outputs."""
        # Check if using HF wrapper
        is_hf = hasattr(llm, 'model') and hasattr(llm, 'processor')
        
        if not is_hf:
            if not self._warned:
                print("[warn] Native encoder embeddings only supported with HF backend (--use-hf)")
                self._warned = True
            return {}
        
        try:
            # Load image
            image = self._load_image(sample)
            if image is None:
                return {}
            
            # Get model and check architecture
            model = llm.model
            processor = llm.processor
            
            # Create cache key for this sample
            cache_key = f"{sample.dataset_id}_{sample.image_id}"
            
            # Extract embeddings based on model type and cache intermediate outputs
            if hasattr(llm, 'is_internvl') and llm.is_internvl:
                embedding, encoder_outputs = self._extract_internvl_encoder_cached(
                    model, processor, image, sample.prompt
                )
            else:
                embedding, encoder_outputs = self._extract_qwen_encoder_cached(
                    model, processor, image, sample.prompt
                )
            
            if embedding is None:
                return {}
            
            # Store encoder outputs in both hook and wrapper for reuse
            if encoder_outputs is not None:
                self._cached_encoder_outputs[cache_key] = encoder_outputs
                if hasattr(llm, '_cached_encoder_outputs'):
                    llm._cached_encoder_outputs[cache_key] = encoder_outputs
            
            return {self.layer_name: embedding}
            
        except Exception as exc:
            if not self._warned:
                print(f"[warn] Native encoder embedding extraction failed: {exc}")
                self._warned = True
            return {}
    
    def _load_image(self, sample: Any) -> Image.Image | None:
        """Load image from sample."""
        image_path = getattr(sample, 'image_path', None)
        if not image_path:
            return None
        try:
            return Image.open(image_path).convert('RGB')
        except Exception:
            return None
    
    def _extract_qwen_encoder_cached(
        self, 
        model: Any, 
        processor: Any, 
        image: Image.Image,
        prompt: str
    ) -> tuple[np.ndarray | None, Any]:
        """Extract vision encoder embeddings from Qwen3-VL model and return cached outputs."""
        try:
            # Try to access Qwen's visual/vision encoder directly
            visual_encoder = None
            if hasattr(model, 'visual'):
                visual_encoder = model.visual
            elif hasattr(model, 'vision_model'):
                visual_encoder = model.vision_model
            elif hasattr(model, 'vision_tower'):
                visual_encoder = model.vision_tower
            
            if visual_encoder is not None:
                # Extract pure vision embeddings directly from vision encoder
                inputs = processor(
                    text="",  # Empty text to get just image processing
                    images=image,
                    return_tensors="pt"
                ).to(model.device)
                
                with torch.no_grad():
                    # Get vision features from the visual encoder
                    # Qwen3-VL requires grid_thw parameter and returns (image_embeds, deepstack_embeds)
                    if 'image_grid_thw' in inputs:
                        vision_outputs = visual_encoder(
                            inputs['pixel_values'],
                            grid_thw=inputs['image_grid_thw']
                        )
                    elif 'pixel_values' in inputs:
                        vision_outputs = visual_encoder(inputs['pixel_values'], output_hidden_states=True, return_dict=True)
                    else:
                        # Fallback to direct call
                        vision_outputs = visual_encoder(inputs['pixel_values'])
                    
                    # Extract hidden states
                    # Qwen vision encoder returns a tuple: (image_embeds, deepstack_embeds)
                    if isinstance(vision_outputs, tuple):
                        # Use the first element (image_embeds)
                        encoder_hidden_states = vision_outputs[0]
                    elif hasattr(vision_outputs, 'last_hidden_state'):
                        encoder_hidden_states = vision_outputs.last_hidden_state
                    elif hasattr(vision_outputs, 'hidden_states') and vision_outputs.hidden_states:
                        encoder_hidden_states = vision_outputs.hidden_states[-1]
                    else:
                        encoder_hidden_states = vision_outputs
                    
                    # Debug: Print actual shape for first extraction
                    if not hasattr(self, '_shape_printed'):
                        print(f"[debug] Qwen vision encoder output shape: {encoder_hidden_states.shape}")
                        if isinstance(vision_outputs, tuple):
                            print(f"[debug] Qwen vision_outputs is tuple with {len(vision_outputs)} elements")
                            for i, elem in enumerate(vision_outputs):
                                if hasattr(elem, 'shape'):
                                    print(f"[debug]   Element {i} shape: {elem.shape}")
                        self._shape_printed = True
                    
                    # Ensure encoder_hidden_states has correct shape [batch, seq, hidden]
                    if encoder_hidden_states.dim() == 2:
                        # Add batch dimension if missing: [seq, hidden] -> [1, seq, hidden]
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
                    
                    # Pool the embeddings for similarity matching
                    pooled = self._pool_embeddings(encoder_hidden_states)
                    
                    # Debug: Print shapes
                    if hasattr(self, '_shape_printed') and not hasattr(self, '_final_printed'):
                        print(f"[debug] Qwen pooled shape: {pooled.shape}")
                        self._final_printed = True
                    
                    # Convert to float32 for numpy compatibility
                    pooled_float = pooled.float() if pooled.dtype == torch.bfloat16 else pooled
                    embedding = pooled_float.cpu().numpy().astype('float32')
                    
                    # Auto-detect dimension (don't assume 2048)
                    actual_dim = embedding.shape[0]
                    if not hasattr(self, '_dim_logged'):
                        print(f"[info] Qwen encoder embedding dimension: {actual_dim}")
                        self._dim_logged = True
                    
                    # Cache vision encoder outputs (not useful for Qwen generation, but kept for consistency)
                    cached_outputs = {
                        'vision_hidden_states': encoder_hidden_states,
                    }
                    return embedding, cached_outputs
            else:
                # Fallback: extract from input embeddings (includes text+vision mixed)
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(model.device)
                
                with torch.no_grad():
                    input_ids = inputs['input_ids']
                    
                    if hasattr(model, 'get_input_embeddings'):
                        embed_layer = model.get_input_embeddings()
                        input_embeds = embed_layer(input_ids)
                        
                        # Pool the embeddings for similarity matching
                        pooled = self._pool_embeddings(input_embeds)
                        pooled_float = pooled.float() if pooled.dtype == torch.bfloat16 else pooled
                        embedding = pooled_float.cpu().numpy().astype('float32')
                        
                        cached_outputs = {
                            'input_embeds': input_embeds,
                            'attention_mask': inputs.get('attention_mask'),
                        }
                        return embedding, cached_outputs
                    else:
                        return None, None
                
        except Exception as exc:
            if not self._warned:
                print(f"[warn] Qwen encoder extraction failed: {exc}")
                self._warned = True
            return None, None
    
    def _extract_qwen_encoder(
        self, 
        model: Any, 
        processor: Any, 
        image: Image.Image,
        prompt: str
    ) -> np.ndarray | None:
        """Extract encoder embeddings from Qwen3-VL model (legacy method)."""
        embedding, _ = self._extract_qwen_encoder_cached(model, processor, image, prompt)
        return embedding
    
    def _extract_internvl_encoder_cached(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        prompt: str
    ) -> tuple[np.ndarray | None, Any]:
        """Extract encoder embeddings from InternVL model and return cached outputs."""
        try:
            # InternVL uses image transform
            if hasattr(model, 'vision_model') or hasattr(model, 'vision_tower'):
                # Get vision tower
                vision_model = getattr(model, 'vision_model', None) or getattr(model, 'vision_tower', None)
                if vision_model is None:
                    return None, None
                
                # Transform image
                from torchvision import transforms as T
                from torchvision.transforms.functional import InterpolationMode
                
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
                
                pixel_values = transform(image).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
                
                # Extract vision embeddings
                with torch.no_grad():
                    vision_outputs = vision_model(pixel_values, output_hidden_states=True, return_dict=True)
                    if hasattr(vision_outputs, 'last_hidden_state'):
                        encoder_hidden_states = vision_outputs.last_hidden_state
                    elif hasattr(vision_outputs, 'hidden_states'):
                        encoder_hidden_states = vision_outputs.hidden_states[-1]
                    else:
                        encoder_hidden_states = vision_outputs[0] if isinstance(vision_outputs, tuple) else vision_outputs
                    
                    # Pool the embeddings for similarity matching
                    pooled = self._pool_embeddings(encoder_hidden_states)
                    # Convert to float32 for numpy compatibility
                    pooled_float = pooled.float() if pooled.dtype == torch.bfloat16 else pooled
                    embedding = pooled_float.cpu().numpy().astype('float32')
                    
                    # Auto-detect dimension
                    actual_dim = embedding.shape[0]
                    if not hasattr(self, '_dim_logged'):
                        print(f"[info] InternVL encoder embedding dimension: {actual_dim}")
                        self._dim_logged = True
                    
                    # Cache pixel_values for reuse
                    cached_outputs = {
                        'pixel_values': pixel_values,
                        'vision_hidden_states': encoder_hidden_states,
                    }
                    return embedding, cached_outputs
                    
        except Exception as exc:
            print(f"[warn] InternVL encoder extraction failed: {exc}")
            return None, None
    
    def _extract_internvl_encoder(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        prompt: str
    ) -> np.ndarray | None:
        """Extract encoder embeddings from InternVL model (legacy method)."""
        embedding, _ = self._extract_internvl_encoder_cached(model, processor, image, prompt)
        return embedding
    
    def _pool_embeddings(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool hidden states to a single vector."""
        if self.pooling == "mean":
            return hidden_states.mean(dim=1).squeeze(0)
        elif self.pooling == "first":
            return hidden_states[:, 0, :].squeeze(0)
        elif self.pooling == "last":
            return hidden_states[:, -1, :].squeeze(0)
        else:
            return hidden_states.mean(dim=1).squeeze(0)


class NativeTextEmbeddingHook:
    """
    Extracts text embeddings from the VLM's language model.
    Processes just the text prompt (no image) through the LLM's embedding layer.
    """
    
    def __init__(self, layer_name: str = "native_text", pooling: str = "mean"):
        """
        Args:
            layer_name: Name for this embedding layer in cache
            pooling: How to pool the text embeddings ("mean", "first", "last")
        """
        self.layer_name = layer_name
        self.pooling = pooling
        self._warned = False
    
    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        """Extract text embedding from the VLM's language model."""
        # Check if using HF wrapper
        is_hf = hasattr(llm, 'model') and hasattr(llm, 'processor')
        
        if not is_hf:
            if not self._warned:
                print("[warn] Native text embeddings only supported with HF backend (--use-hf)")
                self._warned = True
            return {}
        
        try:
            model = llm.model
            processor = llm.processor
            
            # Tokenize just the text prompt (no image)
            inputs = processor(
                text=sample.prompt,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                # Get text embeddings from the model's embedding layer
                if hasattr(model, 'get_input_embeddings'):
                    embed_layer = model.get_input_embeddings()
                    text_embeds = embed_layer(inputs['input_ids'])
                    
                    # Pool the embeddings
                    pooled = self._pool_embeddings(text_embeds)
                    pooled_float = pooled.float() if pooled.dtype == torch.bfloat16 else pooled
                    embedding = pooled_float.cpu().numpy().astype('float32')
                    
                    return {self.layer_name: embedding}
                else:
                    return {}
                    
        except Exception as exc:
            if not self._warned:
                print(f"[warn] Native text embedding extraction failed: {exc}")
                self._warned = True
            return {}
    
    def _pool_embeddings(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool hidden states to a single vector."""
        if self.pooling == "mean":
            return hidden_states.mean(dim=1).squeeze(0)
        elif self.pooling == "first":
            return hidden_states[:, 0, :].squeeze(0)
        elif self.pooling == "last":
            return hidden_states[:, -1, :].squeeze(0)
        else:
            return hidden_states.mean(dim=1).squeeze(0)


class NativeDecoderEmbeddingHook:
    """
    Extracts embeddings from the VLM's decoder/LLM part after processing the prompt.
    Uses the language model's representation of the multimodal input.
    """
    
    def __init__(self, layer_name: str = "native_decoder", pooling: str = "mean"):
        """
        Args:
            layer_name: Name for this embedding layer in cache
            pooling: How to pool the decoder outputs ("mean", "first", "last")
        """
        self.layer_name = layer_name
        self.pooling = pooling
        self._warned = False
    
    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        """Extract decoder embedding from the model."""
        # Check if using HF wrapper
        is_hf = hasattr(llm, 'model') and hasattr(llm, 'processor')
        
        if not is_hf:
            if not self._warned:
                print("[warn] Native decoder embeddings only supported with HF backend (--use-hf)")
                self._warned = True
            return {}
        
        try:
            # Load image
            image = self._load_image(sample)
            if image is None:
                return {}
            
            # Get model and processor
            model = llm.model
            processor = llm.processor
            
            # Extract embeddings based on model type
            if hasattr(llm, 'is_internvl') and llm.is_internvl:
                embedding = self._extract_internvl_decoder(model, processor, image, sample.prompt)
            else:
                embedding = self._extract_qwen_decoder(model, processor, image, sample.prompt)
            
            if embedding is None:
                return {}
            
            return {self.layer_name: embedding}
            
        except Exception as exc:
            if not self._warned:
                print(f"[warn] Native decoder embedding extraction failed: {exc}")
                self._warned = True
            return {}
    
    def _load_image(self, sample: Any) -> Image.Image | None:
        """Load image from sample."""
        image_path = getattr(sample, 'image_path', None)
        if not image_path:
            return None
        try:
            return Image.open(image_path).convert('RGB')
        except Exception:
            return None
    
    def _extract_qwen_decoder(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        prompt: str
    ) -> np.ndarray | None:
        """Extract decoder embeddings from Qwen3-VL model."""
        try:
            # Process inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)
            
            # Get input embeddings then run through initial layers
            with torch.no_grad():
                # Forward pass with hidden states
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                
                # Get last layer hidden states (decoder representations)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    decoder_hidden_states = outputs.hidden_states[-1]  # Last layer
                elif hasattr(outputs, 'last_hidden_state'):
                    decoder_hidden_states = outputs.last_hidden_state
                else:
                    return None
                
                # Pool the embeddings
                pooled = self._pool_embeddings(decoder_hidden_states)
                # Convert to float32 for numpy compatibility
                pooled_float = pooled.float() if pooled.dtype == torch.bfloat16 else pooled
                return pooled_float.cpu().numpy().astype('float32')
                
        except Exception as exc:
            print(f"[warn] Qwen decoder extraction failed: {exc}")
            return None
    
    def _extract_internvl_decoder(
        self,
        model: Any,
        processor: Any,
        image: Image.Image,
        prompt: str
    ) -> np.ndarray | None:
        """Extract decoder embeddings from InternVL model."""
        try:
            # For InternVL, we need to get the LLM hidden states
            # This requires running through the model's encode path
            
            # Transform image
            from torchvision import transforms as T
            from torchvision.transforms.functional import InterpolationMode
            
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            
            pixel_values = transform(image).unsqueeze(0).to(model.device, dtype=torch.bfloat16)
            
            # Tokenize prompt
            input_ids = processor(prompt, return_tensors='pt')['input_ids'].to(model.device)
            
            # Get language model
            if hasattr(model, 'language_model') or hasattr(model, 'llm'):
                llm_model = getattr(model, 'language_model', None) or getattr(model, 'llm', None)
                
                # Extract embeddings through the multimodal encoding
                with torch.no_grad():
                    # Get vision embeddings first
                    if hasattr(model, 'extract_feature'):
                        img_embeds = model.extract_feature(pixel_values)
                    elif hasattr(model, 'encode_images'):
                        img_embeds = model.encode_images(pixel_values)
                    else:
                        # Fallback: just use LLM input embeddings
                        outputs = llm_model(input_ids, output_hidden_states=True, return_dict=True)
                        decoder_hidden_states = outputs.hidden_states[-1]
                        pooled = self._pool_embeddings(decoder_hidden_states)
                        pooled_float = pooled.float() if pooled.dtype == torch.bfloat16 else pooled
                        return pooled_float.cpu().numpy().astype('float32')
                    
                    # Get text embeddings
                    text_embeds = llm_model.get_input_embeddings()(input_ids)
                    
                    # Combine (simplified - actual InternVL does more complex fusion)
                    # Use text embeddings as decoder representation
                    pooled = self._pool_embeddings(text_embeds)
                    pooled_float = pooled.float() if pooled.dtype == torch.bfloat16 else pooled
                    return pooled_float.cpu().numpy().astype('float32')
            
            return None
                    
        except Exception as exc:
            print(f"[warn] InternVL decoder extraction failed: {exc}")
            return None
    
    def _pool_embeddings(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool hidden states to a single vector."""
        if self.pooling == "mean":
            return hidden_states.mean(dim=1).squeeze(0)
        elif self.pooling == "first":
            return hidden_states[:, 0, :].squeeze(0)
        elif self.pooling == "last":
            return hidden_states[:, -1, :].squeeze(0)
        else:
            return hidden_states.mean(dim=1).squeeze(0)


class NativeTextVisionEmbeddingHook:
    """
    Combines native text and vision encoder embeddings from the VLM.
    Provides both text and vision representations from the model's native encoders.
    """
    
    def __init__(self, pooling: str = "mean"):
        self.text_hook = NativeTextEmbeddingHook(layer_name="native_text", pooling=pooling)
        self.vision_hook = NativeEncoderEmbeddingHook(layer_name="native_encoder", pooling=pooling)
    
    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        """Extract both text and vision encoder embeddings."""
        payload: dict[str, np.ndarray] = {}
        payload.update(self.text_hook(llm=llm, sample=sample))
        payload.update(self.vision_hook(llm=llm, sample=sample))
        return payload


class NativeEncoderDecoderEmbeddingHook:
    """
    Combines both encoder and decoder embeddings from the VLM.
    Provides both vision and language representations.
    """
    
    def __init__(self, pooling: str = "mean"):
        self.encoder_hook = NativeEncoderEmbeddingHook(layer_name="native_encoder", pooling=pooling)
        self.decoder_hook = NativeDecoderEmbeddingHook(layer_name="native_decoder", pooling=pooling)
    
    def __call__(self, *, llm: Any, sample: Any) -> dict[str, np.ndarray]:
        """Extract both encoder and decoder embeddings."""
        payload: dict[str, np.ndarray] = {}
        payload.update(self.encoder_hook(llm=llm, sample=sample))
        payload.update(self.decoder_hook(llm=llm, sample=sample))
        return payload
