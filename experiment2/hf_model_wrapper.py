"""
Hugging Face Transformers wrapper that's compatible with vLLM-style API.
This allows running experiments without vLLM's buggy multimodal cache.
"""
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, Qwen3VLForConditionalGeneration, AutoModel, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


class HFModelWrapper:
    """Wrapper that mimics vLLM's LLM interface using HuggingFace transformers."""
    
    def __init__(
        self,
        model: str,
        trust_remote_code: bool = True,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        **kwargs
    ):
        self.model_name = model
        # Cache encoder outputs to avoid recomputation when using native embeddings
        self._cached_encoder_outputs: dict[str, Any] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_internvl = "InternVL" in model
        
        print(f"[HF] Loading model {model} on {self.device}...")
        
        # For InternVL, load tokenizer separately; for others, use processor
        if self.is_internvl:
            self.processor = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=trust_remote_code
            )
            # InternVL image preprocessing
            self.image_transform = self._build_internvl_transform()
        else:
            self.processor = AutoProcessor.from_pretrained(
                model,
                trust_remote_code=trust_remote_code
            )
        
        # Use model-specific class if available
        if "Qwen3-VL" in model or "Qwen3VL" in model:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=trust_remote_code
            )
        elif "InternVL" in model:
            # InternVL requires AutoModel with trust_remote_code
            self.model = AutoModel.from_pretrained(
                model,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=trust_remote_code
            )
            # Set img_context_token_id for InternVL models
            img_context_token_id = self.processor.convert_tokens_to_ids('<IMG_CONTEXT>')
            self.model.img_context_token_id = img_context_token_id
            print(f"[HF] Set img_context_token_id = {img_context_token_id}")
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=trust_remote_code
            )
        
        if self.device == "cpu":
            print("[HF] Warning: Running on CPU, will be slow")
        
        print(f"[HF] Model loaded successfully")
    
    def _build_internvl_transform(self, input_size=448):
        """Build image transform for InternVL models."""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Any,
        images: Optional[List[Image.Image]] = None,
        encoder_cache_key: Optional[str] = None,
    ) -> List[Any]:
        """
        Generate responses for prompts.
        
        Args:
            prompts: List of text prompts
            sampling_params: SamplingParams object with temperature, max_tokens, etc.
            images: Optional list of PIL images (one per prompt)
            encoder_cache_key: Optional key to reuse cached encoder outputs
        
        Returns:
            List of output objects compatible with vLLM's RequestOutput
        """
        outputs = []
        
        for i, prompt in enumerate(prompts):
            image = images[i] if images and i < len(images) else None
            
            if self.is_internvl:
                # InternVL uses special chat method
                response_text = self._generate_internvl(prompt, image, sampling_params, encoder_cache_key)
            else:
                # Standard HF pipeline
                response_text = self._generate_standard(prompt, image, sampling_params, encoder_cache_key)
            
            # Create output object compatible with vLLM
            output = HFOutput(
                prompt=prompt,
                outputs=[HFCompletionOutput(text=response_text)],
                finished=True
            )
            outputs.append(output)
        
        return outputs
    
    def _generate_internvl(self, prompt: str, image: Optional[Image.Image], sampling_params: Any, encoder_cache_key: Optional[str] = None) -> str:
        """Generate using InternVL's chat method."""
        if image is None:
            # Text-only not supported well by InternVL chat, use dummy image
            image = Image.new('RGB', (448, 448), color='white')
        
        # Try to reuse cached encoder outputs
        pixel_values = None
        if encoder_cache_key and encoder_cache_key in self._cached_encoder_outputs:
            cached = self._cached_encoder_outputs[encoder_cache_key]
            pixel_values = cached.get('pixel_values')
        
        # If not cached, preprocess image
        if pixel_values is None:
            pixel_values = self.image_transform(image).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        
        # Create generation config
        generation_config = {
            'max_new_tokens': sampling_params.max_tokens,
            'do_sample': sampling_params.temperature > 0,
        }
        if sampling_params.temperature > 0:
            generation_config['temperature'] = sampling_params.temperature
        
        # Use chat method
        with torch.no_grad():
            response = self.model.chat(
                tokenizer=self.processor,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=generation_config
            )
        
        return response
    
    def _generate_standard(self, prompt: str, image: Optional[Image.Image], sampling_params: Any, encoder_cache_key: Optional[str] = None) -> str:
        """Generate using standard HF pipeline."""
        # NOTE: Caching input_embeds doesn't work well for Qwen because the model
        # expects to process images internally during generate(). Passing pre-computed
        # input_embeds causes "Image features and image tokens do not match" error.
        # So we always reprocess for Qwen models.
        
        # Prepare inputs
        if image:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
        
        # Generate
        with torch.no_grad():
            # Standard generation with input_ids
            # (We can't use cached inputs_embeds for Qwen - causes image token mismatch)
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature if sampling_params.temperature > 0 else 1.0,
                do_sample=sampling_params.temperature > 0,
            )
            # Skip the input tokens for models that include them
            response_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response_text = self.processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response_text


class HFOutput:
    """Mimics vLLM's RequestOutput."""
    def __init__(self, prompt: str, outputs: List, finished: bool):
        self.prompt = prompt
        self.outputs = outputs
        self.finished = finished


class HFCompletionOutput:
    """Mimics vLLM's CompletionOutput."""
    def __init__(self, text: str):
        self.text = text
