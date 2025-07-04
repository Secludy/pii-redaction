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

    def _model_call(self, raw_text_input, model_index):
        """
        Process text through a specific model to identify PII entities.

        Args:
            raw_text_input (str or list[str]): The raw text(s) to process.
            model_index (int): Index of the model config to use.

        Returns:
            str or list[str]: The processed text(s) with PII tags, matching input type.
        """
        self._initialize_model(model_index) # Ensure the correct model is loaded

        model_config = self._get_model_config(model_index)
        model_path = model_config["path"]
        configured_max_new_tokens = model_config.get("max_new_tokens", 1024)

        if self.engine == "transformers":
            if not isinstance(raw_text_input, str):
                raise TypeError(f"Transformers engine expects a single string for raw_text_input, got {type(raw_text_input)}")
        
            messages = [{"role": "user", "content": raw_text_input}]
            if model_path not in self.loaded_models or self.loaded_models[model_path] is None:
                raise RuntimeError(f"Transformers model {model_path} not initialized before _model_call.")
        
            model_data = self.loaded_models[model_path]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
        
            try:
                encoded_input = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
            except Exception as e:
                print(f"Warning: tokenizer.apply_chat_template with add_generation_prompt failed: {e}. Falling back.")
                prompt_str = tokenizer.decode(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
                encoded_input = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=getattr(tokenizer, 'model_max_length', 2048))

            input_ids = encoded_input["input_ids"].to(model.device)
            attention_mask = encoded_input.get("attention_mask") 
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
        
            prompt_token_length = input_ids.shape[1]
            effective_max_len = getattr(tokenizer, 'model_max_length', 2048)

            if prompt_token_length >= effective_max_len:
                print(f"Warning: Input length ({prompt_token_length}) after chat template potentially exceeds model max length ({effective_max_len}). Output might be compromised.")

            allowable_max_new = max(0, effective_max_len - prompt_token_length - 5)
            actual_max_new_tokens = min(configured_max_new_tokens, allowable_max_new)

            if actual_max_new_tokens <= 0 and prompt_token_length < effective_max_len:
                 actual_max_new_tokens = configured_max_new_tokens 
            elif actual_max_new_tokens <= 0:
                print(f"Warning: No space for new tokens. Prompt length {prompt_token_length}, model max {effective_max_len}. Returning empty string.")
                return "" 

            with torch.no_grad():
                outputs_tensor = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=actual_max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                    temperature=0.0, top_p=1.0, num_beams=1
                )
        
            generated_ids = outputs_tensor[0][input_ids.shape[1]:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        elif self.engine == "vllm":
            if self.llm is None: 
                 raise RuntimeError("vLLM engine not properly initialized before generation.")
        
            is_batch = isinstance(raw_text_input, list)
            texts_to_batch = raw_text_input if is_batch else [raw_text_input]

            try:
                tokenizer = self.llm.get_tokenizer()
            except Exception as e:
                print(f"Warning: self.llm.get_tokenizer() failed: {e}. Attempting manual load for vLLM prompt formatting.")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                except Exception as e_load:
                    raise RuntimeError(f"Failed to load tokenizer for vLLM ({model_path}): {e_load}")

            prompt_strings = []
            for text_item in texts_to_batch:
                messages = [{"role": "user", "content": text_item}]
                prompt_strings.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
        
            sampling_params = SamplingParams(
                max_tokens=configured_max_new_tokens,
                temperature=0.0, 
                top_p=1.0
            )
        
            vllm_batch_outputs = self.llm.generate(prompt_strings, sampling_params)
        
            results = [output_item.outputs[0].text.strip() for output_item in vllm_batch_outputs]
        
            return results if is_batch else results[0]
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

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
        for model_idx, model_config in enumerate(tqdm(self.models, desc="Processing Models")):
            self._initialize_model(model_idx) # Load/ensure current model is ready
            model_name = model_config.get("path", f"Model {model_idx+1}")
            
            # Determine tags_to_include_for_apply for this model once
            if model_config.get("strict_apply_filter", True):
                model_tags_for_apply_config.append(model_config.get("tags", []))
            else:
                model_tags_for_apply_config.append(None) 

            texts_to_process_for_current_model = []
            doc_indices_for_current_batch = [] # Keep track of original doc indices

            for doc_idx, doc_text in enumerate(documents):
                current_doc_text_for_processing = doc_text
                
                if self.engine == "vllm":
                    texts_to_process_for_current_model.append(current_doc_text_for_processing)
                    doc_indices_for_current_batch.append(doc_idx)
                else: # transformers engine, process one by one (add tqdm here for user feedback)
                    if doc_idx == 0: # Setup tqdm for transformers on first doc for this model
                        print(f"Processing documents with Transformers model: {model_name}")
                        transformer_doc_iterator = tqdm(enumerate(documents), total=num_documents, desc=f"Docs for {model_name}")
                    
                    # This loop structure for transformers needs adjustment.
                    # The current outer loop is by model, then by document for collecting texts_to_process.
                    # For transformers, we should process here directly.
                    # The refactor below will adjust this section.
                    pass # Placeholder, will be addressed in the next section of this edit
            
            # Process collected texts
            if self.engine == "vllm":
                if texts_to_process_for_current_model:
                    print(f"Submitting batch of {len(texts_to_process_for_current_model)} documents to vLLM model: {model_name}")
                    batch_results = self._model_call(texts_to_process_for_current_model, model_idx)
                    for i, tagged_text in enumerate(batch_results):
                        original_doc_idx = doc_indices_for_current_batch[i]
                        all_doc_raw_outputs[original_doc_idx][model_idx] = tagged_text
            elif self.engine == "transformers":
                # Correct processing loop for transformers within the model-first iteration
                print(f"Processing documents with Transformers model: {model_name}")
                for doc_idx, doc_text in tqdm(enumerate(documents), total=num_documents, desc=f"Docs for {model_name}"):
                    current_doc_text_for_processing = doc_text # Each model sees original text
                    tagged_text = self._model_call(current_doc_text_for_processing, model_idx)
                    all_doc_raw_outputs[doc_idx][model_idx] = tagged_text

            if self.engine == "transformers":
                # Unload transformers model if not needed by next model_config (if paths differ)
                # This specific call might be too aggressive if models share paths or for overall efficiency.
                # Consider unloading only if PIIRedactor instance is done or if memory is critical.
                # For now, matching previous logic, but _unload_model(unload_all=True) at the end is more robust.
                self._unload_model(model_idx) 
{{ ... }}
