import torch
import re
import json
# Ensure vllm is optional for users who only use transformers engine
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None 
    # print("Warning: vLLM not installed. vLLM engine will not be available.")

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from collections import defaultdict
import difflib
from enum import Enum
from .faker_utils import FakePIIGenerator
import gc # For garbage collection, potentially useful with vLLM model switching


class PIIHandlingMode(Enum):
    """Enum for different PII handling modes"""

    TAG = "tag"  # Keep PII content between XML tags: <PII:type>content</PII:type>
    REDACT = "redact"  # Replace PII with just an empty tag: <PII:type/>
    REPLACE = "replace"  # Replace PII with fake data: <PII:type>fake_data</PII:type>


class PIIType(Enum):
    """Enum for different PII types that can be identified and redacted"""

    AGE = "age"  # A person's age
    CREDIT_CARD_INFO = (
        "credit_card_info"  # A credit card number, expiration date, CCV, etc.
    )
    NATIONALITY = "nationality"  # A country when used to reference place of birth, residence, or citizenship
    DATE = "date"  # A specific calendar date
    DATE_OF_BIRTH = "date_of_birth"  # A specific calendar date representing birth
    DOMAIN_NAME = "domain_name"  # A domain on the internet
    EMAIL_ADDRESS = "email_address"  # An email ID
    DEMOGRAPHIC_GROUP = (
        "demographic_group"  # Anything that identifies race or ethnicity
    )
    GENDER = "gender"  # A gender identifier
    PERSONAL_ID = (
        "personal_id"  # Any ID string like a national ID, subscriber number, etc.
    )
    OTHER_ID = "other_id"  # Any ID not associated with a person like an organization ID, database ID, etc.
    BANKING_NUMBER = "banking_number"  # A number associated with a bank account
    MEDICAL_CONDITION = "medical_condition"  # A diagnosis, treatment code or other information identifying a medical condition
    ORGANIZATION_NAME = "organization_name"  # Name of an organization
    PERSON_NAME = "person_name"  # Name of a person
    PHONE_NUMBER = "phone_number"  # A telephone number
    STREET_ADDRESS = "street_address"  # A physical address
    PASSWORD = "password"  # A secure string used for authentication
    SECURE_CREDENTIAL = "secure_credential"  # Any secure credential like an API key, private key, 2FA token
    RELIGIOUS_AFFILIATION = (
        "religious_affiliation"  # Anything that identifies religious affiliation
    )
    LOCATION = "location" # Added LOCATION if it was missing, ensure it's defined once
    IP_ADDRESS = "ip_address" # Added IP_ADDRESS if it was missing, ensure it's defined once

# Ensure LOCATION and IP_ADDRESS are not duplicated if they were already present above.
# The primary definition should be kept. This is a safeguard.
# For example, if LOCATION = "location" is already defined above, these lines would be redundant.
# However, to be safe for the edit tool, we list them here. A manual check might be needed if errors occur.
# A better approach is to ensure the enum is complete and correct once.


def parse_tagged_string(tagged_str):
    """
    Parses a tagged string (with PII tags) and returns a tuple (clean_str, annotations) where:
      - clean_str is the string with all tags removed.
      - annotations is a list of tuples (start, end, tag, annotated_text) for each annotated span.
    """
    annotations = []
    clean_str = ""
    i = 0
    clean_index = 0
    open_tag_pattern = re.compile(r"<PII:(\w+)>")

    while i < len(tagged_str):
        if tagged_str[i] == "<":
            m = open_tag_pattern.match(tagged_str, i)
            if m:
                tag = m.group(1)
                annotation_start = clean_index
                i = m.end()
                closing_tag = f"</PII:{tag}>"
                closing_index = tagged_str.find(closing_tag, i)

                if closing_index == -1:
                    if i < len(tagged_str): 
                        clean_str += tagged_str[i]
                        clean_index += 1
                        i += 1
                    continue

                annotated_text = tagged_str[i:closing_index]
                annotations.append(
                    (
                        annotation_start,
                        annotation_start + len(annotated_text),
                        tag,
                        annotated_text,
                    )
                )
                clean_str += annotated_text
                clean_index += len(annotated_text)
                i = closing_index + len(closing_tag)
            else:
                clean_str += tagged_str[i]
                clean_index += 1
                i += 1
        else:
            clean_str += tagged_str[i]
            clean_index += 1
            i += 1
    return clean_str, annotations


def find_best_match(sub, original, start_hint, window=50):
    search_start = max(0, start_hint - window)
    pos = original.find(sub, search_start)
    if pos != -1:
        return pos

    best_ratio = 0.0
    best_index = -1
    search_end = min(len(original) - len(sub) + 1, start_hint + window)
    for i in range(search_start, search_end):
        candidate = original[i : i + len(sub)]
        ratio = difflib.SequenceMatcher(None, sub, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = i
    if best_ratio < 0.6:
        return -1
    return best_index


def merge_overlapping_spans(annotations):
    if not annotations:
        return []

    annotations.sort(key=lambda x: x[0])

    merged = []

    group_start, group_end, group_tag = annotations[0]
    best_length = group_end - group_start

    for ann in annotations[1:]:
        start, end, tag = ann

        if start <= group_end:
            group_end = max(group_end, end)
            length = end - start
            if length > best_length:
                best_length = length
                group_tag = tag
        else:
            merged.append((group_start, group_end, group_tag))
            group_start, group_end, group_tag = start, end, tag
            best_length = end - start

    merged.append((group_start, group_end, group_tag))
    return merged


def apply_tags(
    original, tagged_strings, tags_to_include, mode=PIIHandlingMode.TAG, locale="en_US"
):
    candidate_annotations = []

    for tstr, include_tags in zip(tagged_strings, tags_to_include):
        cleaned, annotations = parse_tagged_string(tstr)
        for ann_start, ann_end, tag, text in annotations: 
            if include_tags is not None and tag not in [pii_type.value for pii_type in include_tags]:
                continue 

            rel = ann_start / len(cleaned) if cleaned else 0 
            start_hint = int(rel * len(original))
            orig_start = find_best_match(text, original, start_hint)
            if orig_start == -1:
                continue
            orig_end = orig_start + len(text)
            candidate_annotations.append(
                (
                    orig_start,
                    orig_end,
                    tag,
                    text,
                )
            )

    if mode == PIIHandlingMode.REPLACE:
        fake_generator = FakePIIGenerator(locale=locale)

    merge_input = [(start, end, tag) for start, end, tag, _ in candidate_annotations]

    merged_annotations = merge_overlapping_spans(merge_input)

    merged_with_text = []
    for start, end, tag in merged_annotations:
        original_text = original[start:end]
        merged_with_text.append((start, end, tag, original_text))

    inserts = {}

    for start, end, tag, text in merged_with_text:
        if mode == PIIHandlingMode.TAG:
            inserts[start] = f"<PII:{tag}>{text}</PII:{tag}>"
            for i in range(start + 1, end):
                inserts[i] = ""
        elif mode == PIIHandlingMode.REDACT:
            inserts[start] = f"<PII:{tag}/>"
            for i in range(start + 1, end):
                inserts[i] = ""
        elif mode == PIIHandlingMode.REPLACE:
            fake_value = fake_generator.get_fake_value(tag, text)
            inserts[start] = f"{fake_value}"
            for i in range(start + 1, end):
                inserts[i] = ""

    result = []
    i = 0
    while i <= len(original):
        if i in inserts:
            result.append(inserts[i])
        elif i < len(original):
            result.append(original[i])
        i += 1

    return "".join(result)


class PIIRedactor:
    def __init__(self, device=None, engine="transformers"):
        if engine == "vllm" and LLM is None:
            raise ImportError(
                "vLLM engine selected, but vllm package not installed or failed to import."
            )
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and engine == "vllm" and not torch.cuda.is_available():
            print("Warning: CUDA specified but not available. vLLM might not work as expected.")
        
        self.engine = engine
        self.loaded_models = {}  # For transformers engine: stores {model_path: {'model': model, 'tokenizer': tokenizer}}
        self.llm = None  # For vLLM engine: stores the LLM instance
        self.current_vllm_model_path = None # Tracks the currently loaded vLLM model path
        self.default_vllm_max_new_tokens = 1024 # Default for vLLM if not specified per model

        # Define model configurations
        # Each model is a dictionary with 'path', 'tags' (list of PII types it's good at),
        # 'prompt_template_format_string', 'max_new_tokens' (optional, for vLLM mainly),
        # and 'exclusive_tags' (boolean, if True, its tags won't be processed by subsequent models for a given document part)
        # 'dtype' is also model-specific for precision control (e.g., torch.bfloat16)
        self.models = [
            {
                "path": "OpenPipe/Pii-Redact-Name", 
                "tags": [
                    PIIType.PERSON_NAME,
                    PIIType.ORGANIZATION_NAME,
                ],
                "prompt_template_format_string": "<SYS>You are a PII detection expert. Identify the following PII types: {tags_to_identify}. Output only the PII entities, each on a new line, enclosed in XML tags corresponding to their PII type. If no PII is found, output 'NO_PII_FOUND'.</SYS>\nText to process: {text_to_process}",
                "max_new_tokens": 1024, 
                "exclusive_tags": False, 
                "strict_apply_filter": True,
                "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32 
            },
            {
                "path": "OpenPipe/Pii-Redact-General", 
                "tags": [
                    PIIType.STREET_ADDRESS,
                    PIIType.PHONE_NUMBER,
                    PIIType.EMAIL_ADDRESS,
                    PIIType.PERSONAL_ID,
                    PIIType.OTHER_ID,
                    PIIType.CREDIT_CARD_INFO,
                    PIIType.DATE_OF_BIRTH,
                    PIIType.BANKING_NUMBER,
                    PIIType.MEDICAL_CONDITION,
                    PIIType.PASSWORD,
                    PIIType.SECURE_CREDENTIAL,
                    PIIType.RELIGIOUS_AFFILIATION
                ],
                "prompt_template_format_string": "<SYS>You are a PII detection expert. Identify the following PII types: {tags_to_identify}. Output only the PII entities, each on a new line, enclosed in XML tags corresponding to their PII type. If no PII is found, output 'NO_PII_FOUND'.</SYS>\nText to process: {text_to_process}",
                "max_new_tokens": 1024, # Can be adjusted if general model needs more/less
                "exclusive_tags": False, 
                "strict_apply_filter": False,
                "dtype": torch.bfloat16 if self.device == "cuda" else torch.float32 
            }
        ]

        # Validate that all tags in model configs are valid PIIType members
        for model_config in self.models:
            if not isinstance(model_config.get("tags"), list) or not all(
                isinstance(tag, PIIType) for tag in model_config.get("tags", [])
            ):
                raise ValueError(
                    f"Model configuration for {model_config.get('path')} has invalid 'tags'. "
                    f"Must be a list of PIIType members."
                )
            if not isinstance(model_config.get("prompt_template_format_string"), str):
                 raise ValueError(
                    f"Model configuration for {model_config.get('path')} requires 'prompt_template_format_string' as a string."
                )
            if not isinstance(model_config.get("path"), str):
                 raise ValueError(
                    f"Model configuration requires 'path' as a string."
                )

    def _get_model_config(self, model_index):
        if 0 <= model_index < len(self.models):
            return self.models[model_index]
        raise ValueError(f"Model index {model_index} is out of range.")

    def _initialize_model(self, model_index):
        model_config = self._get_model_config(model_index)
        model_path = model_config["path"]
        model_dtype = model_config.get("dtype", torch.float32 if self.engine == "transformers" else None) 

        if self.engine == "transformers":
            if model_path not in self.loaded_models:
                print(f"Initializing Hugging Face model: {model_path} on {self.device}")
                # Determine torch_dtype for transformers
                torch_dtype_for_transformers = model_dtype if model_dtype else torch.float32
                if self.device == "cuda" and torch_dtype_for_transformers == torch.float32 and torch.cuda.is_bf16_supported():
                    # Prefer bfloat16 on CUDA if float32 is specified but bfloat16 is available and supported
                    # This is a common optimization, but can be made more explicit in config if needed.
                    torch_dtype_for_transformers = torch.bfloat16
                
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch_dtype_for_transformers, 
                    trust_remote_code=True
                )
                model.to(self.device)
                model.eval() # Set to evaluation mode
                self.loaded_models[model_path] = {"model": model, "tokenizer": tokenizer}
        
        elif self.engine == "vllm":
            if self.llm is None or self.current_vllm_model_path != model_path:
                if self.llm is not None:
                    del self.llm
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    self.llm = None 
                    self.current_vllm_model_path = None

                print(f"Initializing vLLM model: {model_path} on {self.device}")
                
                # Determine dtype_str for vLLM from torch.dtype
                dtype_str = "auto"
                if model_dtype == torch.bfloat16:
                    dtype_str = "bfloat16"
                elif model_dtype == torch.float16:
                    dtype_str = "half"
                elif model_dtype == torch.float32:
                    dtype_str = "float32"
                # vLLM also supports "float" (same as float32) and "auto"

                try:
                    self.llm = LLM(
                        model=model_path,
                        tokenizer=model_path, 
                        tensor_parallel_size=torch.cuda.device_count() if self.device == "cuda" and torch.cuda.is_available() else 1,
                        dtype=dtype_str,
                        trust_remote_code=True,  # Ensure this is True
                        max_model_len=model_config.get("max_model_len") # Optional: if you want to set it explicitly
                    )
                    self.current_vllm_model_path = model_path
                except Exception as e:
                    print(f"Error initializing vLLM model {model_path}: {e}")
                    raise
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def _unload_model(self, model_index=None, unload_all=False):
        if unload_all:
            if self.engine == "transformers":
                for model_path in self.loaded_models:
                    del self.loaded_models[model_path]["model"]
                    del self.loaded_models[model_path]["tokenizer"]
                    self.loaded_models[model_path] = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif self.engine == "vllm":
                if self.llm is not None:
                    del self.llm
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    self.llm = None
                    self.current_vllm_model_path = None
        else:
            if self.engine == "transformers":
                model_config = self._get_model_config(model_index)
                model_path = model_config["path"]
                if model_path in self.loaded_models and self.loaded_models[model_path] is not None:
                    del self.loaded_models[model_path]["model"]
                    del self.loaded_models[model_path]["tokenizer"]
                    self.loaded_models[model_path] = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            elif self.engine == "vllm":
                if self.llm is not None and model_index == 0: 
                    del self.llm
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    self.llm = None
                    self.current_vllm_model_path = None

    def _model_call(self, raw_text_to_process, model_index):
        """
        Process text through a specific model to identify PII entities.

        Args:
            raw_text_to_process (str): The raw text to process.
            model_index (int): Index of the model config to use.

        Returns:
            str: The processed text with PII tags
        """
        self._initialize_model(model_index) # Ensure the correct model is loaded

        model_config = self._get_model_config(model_index)
        model_path = model_config["path"]
        configured_max_new_tokens = model_config.get("max_new_tokens", 1024) # Default to 1024

        messages = [{"role": "user", "content": raw_text_to_process}]

        if self.engine == "transformers":
            if model_path not in self.loaded_models or self.loaded_models[model_path] is None:
                raise RuntimeError(f"Transformers model {model_path} not initialized before _model_call.")
        
            model_data = self.loaded_models[model_path]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
        
            # Apply chat template and tokenize
            try:
                # Some tokenizers might not support add_generation_prompt directly in apply_chat_template
                # or expect it to be handled differently. Standard practice is to include it.
                encoded_input = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True, # Important for generation
                    return_tensors="pt"
                )
            except Exception as e:
                # Fallback or simpler application if the above fails (e.g. older tokenizer versions)
                print(f"Warning: tokenizer.apply_chat_template with add_generation_prompt failed: {e}. Falling back.")
                # This fallback might be model-specific. For now, a common alternative:
                prompt_str = tokenizer.decode(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                encoded_input = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048)

            input_ids = encoded_input["input_ids"].to(model.device)
            # attention_mask might not be returned by apply_chat_template if tokenize=True for some versions/models
            # or it might be part of encoded_input. If not, generate() can often infer it for causal LMs.
            attention_mask = encoded_input.get("attention_mask") 
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
        
            # Ensure prompt is not longer than model's capacity - this check is less direct with apply_chat_template
            # as the template itself adds tokens. The tokenizer's truncation (if applied above) should handle it.
            prompt_token_length = input_ids.shape[1]
            effective_max_len = getattr(tokenizer, 'model_max_length', 2048)

            if prompt_token_length >= effective_max_len:
                print(f"Warning: Input length ({prompt_token_length}) after chat template potentially exceeds model max length ({effective_max_len}). Output might be compromised.")
                # actual_max_new_tokens might become 0 or negative if not handled.

            # Use configured_max_new_tokens, ensuring it doesn't exceed model capacity given prompt length
            # This calculation might need adjustment if truncation is handled by tokenizer.apply_chat_template or generate
            allowable_max_new = max(0, effective_max_len - prompt_token_length - 5) # -5 for safety margin
            actual_max_new_tokens = min(configured_max_new_tokens, allowable_max_new)

            if actual_max_new_tokens <= 0 and prompt_token_length < effective_max_len:
                 # If allowable_max_new is zero due to prompt length, but there's technically space
                 actual_max_new_tokens = configured_max_new_tokens # Try with configured, model.generate might truncate if needed
            elif actual_max_new_tokens <= 0:
                print(f"Warning: No space for new tokens. Prompt length {prompt_token_length}, model max {effective_max_len}. Returning empty string.")
                return "" 

            with torch.no_grad():
                outputs_tensor = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask, # Pass attention_mask if available
                    max_new_tokens=actual_max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.0, top_p=1.0, num_beams=1
                )
        
            # The output includes the prompt, so we need to decode only the generated part.
            # For apply_chat_template, the input_ids are the full prompt.
            generated_ids = outputs_tensor[0][input_ids.shape[1]:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        elif self.engine == "vllm":
            if self.llm is None: 
                 raise RuntimeError("vLLM engine not properly initialized before generation.")
        
            # Get tokenizer from vLLM engine
            try:
                tokenizer = self.llm.get_tokenizer()
            except Exception as e:
                # Fallback: try to load it manually if get_tokenizer() fails or not available
                # This might happen if vLLM's LLM object doesn't expose it as expected in all versions/cases
                print(f"Warning: self.llm.get_tokenizer() failed: {e}. Attempting manual load for vLLM prompt formatting.")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                except Exception as e_load:
                    raise RuntimeError(f"Failed to load tokenizer for vLLM ({model_path}): {e_load}")

            # Apply chat template to get the prompt string for vLLM
            prompt_string = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
            sampling_params = SamplingParams(
                max_tokens=configured_max_new_tokens,
                temperature=0.0, 
                top_p=1.0
            )
            vllm_outputs = self.llm.generate([prompt_string], sampling_params)
            return vllm_outputs[0].outputs[0].text.strip()
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

    def tag_pii_in_documents(self, documents, mode=PIIHandlingMode.TAG, locale="en_US"):
        """
        Process a list of documents to identify and handle PII according to the specified mode.
        Args:
            documents (list): List of text documents to process.
            mode (PIIHandlingMode): How to handle identified PII.
            locale (str): Locale for generating fake data.
        Returns:
            list: List of documents with PII handled.
        """
        if not documents:
            return []

        num_documents = len(documents)
        # Initialize a list of lists to store raw model outputs for each document
        # E.g., all_doc_raw_outputs[doc_idx][model_idx] = "tagged_text_from_model"
        all_doc_raw_outputs = [[None] * len(self.models) for _ in range(num_documents)]
        # Store the tags_to_include_for_apply for each model once
        model_tags_for_apply_config = [] 

        # Optimized loop order: iterate through models first, then documents
        # This is especially beneficial for vLLM to avoid frequent model reloads.
        for model_idx, model_config in enumerate(self.models):
            self._initialize_model(model_idx) # Load/ensure current model is ready
            
            # Determine tags_to_include_for_apply for this model once
            if model_config.get("strict_apply_filter", True):
                model_tags_for_apply_config.append(model_config.get("tags", []))
            else:
                model_tags_for_apply_config.append(None) 

            # Prepare texts for batch processing if engine supports it (vLLM does)
            # For transformers, _model_call processes one by one, but model is kept loaded.
            texts_to_process_for_current_model = []
            doc_indices_for_current_batch = [] # Keep track of original doc indices

            for doc_idx, doc_text in enumerate(documents):
                # In the previous version, current_doc_text_for_processing could change if exclusive_tags=True.
                # With the new loop order, each model always sees the original doc_text for simplicity and consistency.
                # If true sequential processing (output of model A becomes input of model B) is needed with this loop order,
                # it would require more complex state management across model iterations.
                # For now, all models operate on the original document text.
                current_doc_text_for_processing = doc_text
                
                if self.engine == "vllm":
                    # For vLLM, collect all texts to process them in a batch with the current model
                    texts_to_process_for_current_model.append(current_doc_text_for_processing)
                    doc_indices_for_current_batch.append(doc_idx)
                else: # transformers engine, process one by one
                    tagged_text = self._model_call(current_doc_text_for_processing, model_idx)
                    all_doc_raw_outputs[doc_idx][model_idx] = tagged_text
            
            # If vLLM, process the batch for the current model
            if self.engine == "vllm" and texts_to_process_for_current_model:
                # _model_call for vLLM needs to be adapted to handle a batch of texts if we want to call it once.
                # Or, we call it multiple times but vLLM internally batches if llm.generate gets a list.
                # The current _model_call expects a single raw_text_to_process.
                # Let's assume _model_call is called per text, and vLLM's llm.generate handles batching if prompts is a list.
                # For simplicity and to reuse _model_call, we'll call it iteratively here.
                # A more optimized vLLM batch call would involve modifying _model_call or llm.generate directly with a list.
                # However, the primary gain is from loading the vLLM model once per model_config.
                for i, text_to_process in enumerate(tqdm(texts_to_process_for_current_model, desc=f"Model {model_idx+1}/{len(self.models)}")):
                    original_doc_idx = doc_indices_for_current_batch[i]
                    tagged_text = self._model_call(text_to_process, model_idx) # model_idx ensures correct vLLM model is used
                    all_doc_raw_outputs[original_doc_idx][model_idx] = tagged_text

            if self.engine == "transformers":
                self._unload_model(model_idx) # Unload transformers model if not needed by next model_config (if paths differ)
                                            # Or better, unload all at the end if PIIRedactor instance is done.
                                            # Current _unload_model unloads by path, so this is okay.

        # After all models have processed all documents, apply tags for each document
        processed_documents = []
        for doc_idx, original_doc_text in enumerate(tqdm(documents, desc="Applying Tags")):
            raw_outputs_for_this_doc = all_doc_raw_outputs[doc_idx]
            
            final_processed_doc = apply_tags(
                original_doc_text,          # Original document text
                raw_outputs_for_this_doc,   # List of raw tagged strings from all models for this doc
                model_tags_for_apply_config,# List of tag lists (or None) for filtering, same for all docs
                mode=mode,
                locale=locale
            )
            processed_documents.append(final_processed_doc)
        
        # Final cleanup of models
        if self.engine == "transformers":
            self._unload_model(unload_all=True) # Unload all transformers models
        elif self.engine == "vllm" and self.llm is not None:
            # The vLLM model (the last one used) is still in self.llm.
            # We can delete it here if the PIIRedactor instance is done.
            # For now, let's assume it's managed by PIIRedactor lifecycle (e.g., in a __del__ or explicit close method)
            # or unloaded if a different vLLM model is initialized next time.
            pass

        return processed_documents


def tag_pii_in_documents(
    documents,
    device=None,
    engine="transformers",
    mode=PIIHandlingMode.TAG,
    locale="en_US",
):
    """
    Convenience function to process a list of documents through a PII tagging model.

    Args:
        documents (list): List of text documents to process.
        device (str): Device to use for processing (e.g., 'cuda', 'cpu').
        engine (str): Backend to use for generation ('transformers' or 'vllm').
        mode (PIIHandlingMode): How to handle identified PII:
            - TAG: Keep PII with XML tags
            - REDACT: Replace PII with empty tags
            - REPLACE: Replace PII with fake data
        locale (str): Locale for generating fake data (only used if mode=REPLACE)

    Returns:
        list: List of documents with PII handled according to the specified mode.
    """
    redactor = PIIRedactor(device=device, engine=engine)
    return redactor.tag_pii_in_documents(documents, mode=mode, locale=locale)


def clean_dataset(
    input_filename,
    output_filename,
    device=None,
    engine="transformers",
    mode=PIIHandlingMode.TAG,
    locale="en_US",
    batch_size: int = 1 # Default to 1 for old behavior, CLI will pass larger for batching
):
    """
    Reads a JSONL dataset and processes the 'content' field in each message.
    Processes JSON objects in batches, updates them with the processed messages,
    and writes them immediately to the output file.

    Args:
        input_filename (str): Path to the input JSONL file.
        output_filename (str): Path to the output JSONL file.
        device (str): Device to use for processing (e.g., 'cuda', 'cpu').
        engine (str): Backend to use for generation ('transformers' or 'vllm').
        mode (PIIHandlingMode): How to handle identified PII.
        locale (str): Locale for generating fake data (only used if mode=REPLACE).
        batch_size (int): Number of JSON objects (lines) to read and process in one batch.
    """
    redactor = PIIRedactor(device=device, engine=engine)

    # Removed initial line counting for tqdm for performance with large files.
    # tqdm will show iteration count and rate without total percentage.
    with open(input_filename, "r") as fin, open(output_filename, "w") as fout:
        json_objs_batch = []
        for line in tqdm(fin, desc="Processing lines"):
            try:
                json_obj = json.loads(line.strip())
                json_objs_batch.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON decode error: {e}. Line: {line.strip()}")
                continue # Skip this line and proceed to the next

            if len(json_objs_batch) >= batch_size:
                process_and_write_batch(
                    json_objs_batch, fout, redactor, mode=mode, locale=locale
                )
                json_objs_batch = []  # Reset batch

        # Process any remaining items in the last batch
        if json_objs_batch:
            process_and_write_batch(
                json_objs_batch, fout, redactor, mode=mode, locale=locale
            )


def process_and_write_batch(
    json_objs_batch, fout, redactor, mode=PIIHandlingMode.TAG, locale="en_US"
):
    """
    Given a batch of JSON objects, extracts all messages, processes them,
    updates the JSON objects, and writes them to the provided output file.

    Args:
        json_objs_batch (list): List of JSON objects.
        fout (file object): Open output file to write processed JSON objects.
        redactor (PIIRedactor): Redactor object to use for tagging.
        mode (PIIHandlingMode): How to handle identified PII:
            - TAG: Keep PII with XML tags
            - REDACT: Replace PII with empty tags
            - REPLACE: Replace PII with fake data
        locale (str): Locale for generating fake data (only used if mode=REPLACE)
    """
    messages_to_process = []
    for obj in json_objs_batch:
        for message in obj.get("messages", []):
            if message["content"]:
                messages_to_process.append(message["content"])

    processed_messages = redactor.tag_pii_in_documents(
        messages_to_process, mode=mode, locale=locale
    )

    msg_idx = 0
    for obj in json_objs_batch:
        if "messages" in obj:
            for i in range(len(obj["messages"])):
                if obj["messages"][i]["content"]:
                    obj["messages"][i]["content"] = processed_messages[msg_idx]
                    msg_idx += 1

        fout.write(json.dumps(obj) + "\n")
    fout.flush()
