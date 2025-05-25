import torch
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from collections import defaultdict
import difflib
from enum import Enum
from typing import List, Union, Dict, Optional

from .faker_utils import FakePIIGenerator

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None

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
            if include_tags != None and tag not in include_tags:
                continue

            rel = ann_start / len(cleaned) if cleaned else 0
            start_hint = int(rel * len(original))
            orig_start = find_best_match(text, original, start_hint)
            if orig_start == -1:
                continue
            orig_end = orig_start + len(text)
            candidate_annotations.append((orig_start, orig_end, tag, text))

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
        """
        Initialize the PIIRedactor with models for PII detection.

        Args:
            device (str): Device to use for inference (e.g., 'cuda', 'cpu')
            engine (str): Backend to use for generation ('transformers' or 'vllm')
        """
        if engine == "vllm" and LLM is None:
            raise ImportError(
                "vLLM engine selected, but vllm package not installed or failed to import. "
                "Please install with `pip install vllm`."
            )

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = engine
        self.loaded_models = {}  # For transformers engine
        self.llm = None  # For vLLM engine, initialized in _initialize_model
        self.default_vllm_max_new_tokens = 1024 # Default max tokens for vLLM

        base_prompt_template = (
            "[INST] You are an expert PII redaction bot. "
            "Your task is to identify and tag PII in the given text.\n"
            "The PII types to identify are: {tags_to_identify}.\n"
            "Wrap each identified PII with XML-like tags, e.g., <PII:PERSON_NAME>John Doe</PII:PERSON_NAME>.\n"
            "Input text:\n{text_to_process}\nTagged text:\n[/INST]"
        )

        self.models = [
            {
                "model_name_or_path": "openpipe/mistral-7b-opi-pii-tagger-v1.1",
                "tags": "all",  # Specifies which PII types this model config targets. 'all' or list of PIIType.value
                "prompt_template_format_string": base_prompt_template,
                "max_new_tokens": 1024,  # Max new tokens for this model (vLLM and Transformers)
                "exclusive_tags": False, # If true, text tagged by this model won't be passed to subsequent models
            }
            # Example for a second model focusing on specific tags if needed:
            # {
            #     "model_name_or_path": "openpipe/mistral-7b-opi-pii-tagger-v1.1", # Or a different model
            #     "tags": [PIIType.CREDIT_CARD_INFO.value, PIIType.BANKING_NUMBER.value],
            #     "prompt_template_format_string": base_prompt_template,
            #     "max_new_tokens": 512,
            #     "exclusive_tags": True,
            # }
        ]
        self.focus_tags = "all"  # Default to all tags, can be changed by user

        if self.engine == "vllm" and self.models:
            tokenizer_path = self.models[0]["model_name_or_path"]
            try:
                hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                self.max_model_len_hint = hf_tokenizer.model_max_length if hasattr(hf_tokenizer, 'model_max_length') and hf_tokenizer.model_max_length else 4096
                del hf_tokenizer
            except Exception as e:
                print(f"Warning: Could not load tokenizer for {tokenizer_path} to determine max_model_len_hint: {e}")
                self.max_model_len_hint = 4096 # Fallback

    def _initialize_model(self, index):
        """Initialize a specific model if it hasn't been already."""
        model_config = self.models[index]
        model_name_or_path = model_config["model_name_or_path"]

        if self.engine == "transformers":
            if model_name_or_path not in self.loaded_models or self.loaded_models[model_name_or_path] is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
                tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
                if self.device:
                    model.to(self.device)
                elif torch.cuda.is_available():
                    model.to("cuda")
                model.eval() # Set to evaluation mode
                self.loaded_models[model_name_or_path] = {"model": model, "tokenizer": tokenizer}
        elif self.engine == "vllm":
            if self.llm is None:
                if LLM is None: # Should have been caught in __init__, but double check
                    raise ImportError("vLLM not installed. Please install with `pip install vllm`.")
                self.llm = LLM(
                    model=model_name_or_path, 
                    trust_remote_code=True, # Added for models like Mistral
                    max_model_len=model_config.get("max_model_len", self.max_model_len_hint) # Use model specific or default hint
                )
                # SamplingParams will be created dynamically in tag_pii_in_documents or _model_call

    def _unload_model(self, index):
        """Unload a specific model to free GPU memory."""
        model_config = self.models[index]
        model_name_or_path = model_config["model_name_or_path"]

        if self.engine == "transformers":
            if model_name_or_path in self.loaded_models and self.loaded_models[model_name_or_path] is not None:
                del self.loaded_models[model_name_or_path]["model"]
                del self.loaded_models[model_name_or_path]["tokenizer"]
                self.loaded_models[model_name_or_path] = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        elif self.engine == "vllm":
            if self.llm is not None and model_name_or_path == self.models[0]["model_name_or_path"]: 
                del self.llm
                self.llm = None
                # self.sampling_params removed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _model_call(self, text, model_index):
        """
        Process text through a specific model to identify PII entities.

        Args:
            text (str): The text to process
            model_index (int): Index of the model config to use.

        Returns:
            str: The processed text with PII tags
        """
        # self._initialize_model(model_index) # Initialization is now handled in tag_pii_in_documents loop
        model_config = self.models[model_index]
        model_name_or_path = model_config["model_name_or_path"]

        # Determine tags_to_identify string for this model_config
        tags_to_identify_list = model_config["tags"]
        if isinstance(tags_to_identify_list, str) and tags_to_identify_list.lower() == "all":
            tags_to_identify_str = ", ".join([pii_type.value for pii_type in PIIType])
        elif isinstance(tags_to_identify_list, list):
            tags_to_identify_str = ", ".join(tags_to_identify_list)
        else:
            print(f"Warning: Invalid 'tags' format in model config {model_index}: {tags_to_identify_list}. Using empty tag list.")
            tags_to_identify_str = ""
        
        prompt = model_config["prompt_template_format_string"].format(
            tags_to_identify=tags_to_identify_str,
            text_to_process=text
        )

        if self.engine == "transformers":
            if model_name_or_path not in self.loaded_models or self.loaded_models[model_name_or_path] is None:
                # This should not happen if _initialize_model was called correctly
                raise RuntimeError(f"Transformers model {model_name_or_path} not initialized before _model_call.")
            
            model_data = self.loaded_models[model_name_or_path]
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]
            
            encoded_input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 2048)
            input_ids = encoded_input["input_ids"].to(model.device)
            attention_mask = encoded_input["attention_mask"].to(model.device)
            
            # Calculate max_new_tokens based on model's max length and prompt length
            # Fallback for max_length if tokenizer.model_max_length is not available
            effective_max_len = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 2048
            # Ensure prompt is not longer than model's capacity
            prompt_token_length = input_ids.shape[1]
            if prompt_token_length >= effective_max_len:
                 # Handle cases where the prompt itself is too long
                print(f"Warning: Prompt length ({prompt_token_length}) exceeds model max length ({effective_max_len}). Truncation occurred. Output might be compromised.")
                # Option: return empty or error, or let the model handle truncated input if possible
                # For now, let it proceed, but be aware of potential issues.
            
            # Use max_new_tokens from model_config, ensuring it doesn't exceed model capacity given prompt length
            configured_max_new = model_config.get("max_new_tokens", 512) # Default if not in config
            allowable_max_new = max(0, effective_max_len - prompt_token_length - 5) # -5 for safety margin
            actual_max_new_tokens = min(configured_max_new, allowable_max_new)

            if actual_max_new_tokens <= 0:
                print(f"Warning: No space for new tokens. Prompt length {prompt_token_length}, model max {effective_max_len}. Returning empty string.")
                return "" # Or handle error appropriately

            with torch.no_grad():
                outputs_tensor = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=actual_max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=0.0, top_p=1.0, num_beams=1 # Added generation params
                )
            
            input_length = input_ids.shape[1]
            generated_ids = outputs_tensor[0][input_length:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        elif self.engine == "vllm":
            if self.llm is None:
                self._initialize_model(model_index) 
                if self.llm is None:
                    raise RuntimeError("vLLM engine not properly initialized before generation.")
            
            current_model_config = self.models[model_index]
            sampling_params = SamplingParams(
                max_tokens=current_model_config.get("max_new_tokens", self.default_vllm_max_new_tokens),
                temperature=0.0, 
                top_p=1.0
            )
            vllm_outputs = self.llm.generate([prompt], sampling_params)
            return vllm_outputs[0].outputs[0].text.strip()
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

    def tag_pii_in_documents(self, documents, mode=PIIHandlingMode.TAG, locale="en_US"):
        """
        Process a list of documents to identify and handle PII according to the specified mode.

        Args:
            documents (list): List of text documents to process.
            mode (PIIHandlingMode): How to handle identified PII:
                - TAG: Keep PII with XML tags
                - REDACT: Replace PII with empty tags
                - REPLACE: Replace PII with fake data
            locale (str): Locale for generating fake data (only used if mode=REPLACE)

        Returns:
            list: List of documents with PII handled according to the specified mode.
        """
        outputs_by_doc = [[] for _ in documents]
        current_doc_texts_for_processing = list(documents)

        for model_idx, model_config in enumerate(self.models):
            self._initialize_model(model_idx)
            next_iteration_doc_texts = ["" for _ in documents]

            if self.engine == "vllm":
                prompts_for_batch = []
                doc_indices_in_batch = []

                tags_to_identify_list = model_config["tags"]
                if isinstance(tags_to_identify_list, str) and tags_to_identify_list.lower() == "all":
                    tags_to_identify_str = ", ".join([pii_type.value for pii_type in PIIType])
                elif isinstance(tags_to_identify_list, list):
                    tags_to_identify_str = ", ".join(tags_to_identify_list)
                else:
                    print(f"Warning: Invalid 'tags' format in model config {model_idx}: {tags_to_identify_list}. Using empty tag list.")
                    tags_to_identify_str = ""

                for doc_idx, text_to_process in enumerate(current_doc_texts_for_processing):
                    if not text_to_process.strip():
                        outputs_by_doc[doc_idx].append((model_idx, ""))
                        next_iteration_doc_texts[doc_idx] = ""
                        continue
                    
                    prompt = model_config["prompt_template_format_string"].format(
                        tags_to_identify=tags_to_identify_str,
                        text_to_process=text_to_process
                    )
                    prompts_for_batch.append(prompt)
                    doc_indices_in_batch.append(doc_idx)

                if prompts_for_batch:
                    if self.llm is None: 
                         raise RuntimeError("vLLM engine not properly initialized before generation.")
                    
                    # Create SamplingParams dynamically using current model_config
                    current_sampling_params = SamplingParams(
                        max_tokens=model_config.get("max_new_tokens", self.default_vllm_max_new_tokens),
                        temperature=0.0,
                        top_p=1.0
                    )
                    vllm_batch_outputs = self.llm.generate(prompts=prompts_for_batch, sampling_params=current_sampling_params)
                    
                    for i, vllm_output_item in enumerate(vllm_batch_outputs):
                        original_doc_idx = doc_indices_in_batch[i]
                        tagged_text = vllm_output_item.outputs[0].text
                        outputs_by_doc[original_doc_idx].append((model_idx, tagged_text))

                        if model_config.get("exclusive_tags"):
                            clean_text, _ = parse_tagged_string(tagged_text)
                            next_iteration_doc_texts[original_doc_idx] = clean_text
                        else:
                            next_iteration_doc_texts[original_doc_idx] = current_doc_texts_for_processing[original_doc_idx]
                
                for i in range(len(documents)):
                    if i not in doc_indices_in_batch:
                        pass 

            elif self.engine == "transformers":
                for doc_idx, text_to_process in enumerate(current_doc_texts_for_processing):
                    if not text_to_process.strip():
                        outputs_by_doc[doc_idx].append((model_idx, ""))
                        next_iteration_doc_texts[doc_idx] = ""
                        continue

                    tagged_text = self._model_call(text_to_process, model_idx)
                    outputs_by_doc[doc_idx].append((model_idx, tagged_text))

                    if model_config.get("exclusive_tags"):
                        clean_text_for_next_model, _ = parse_tagged_string(tagged_text)
                        next_iteration_doc_texts[doc_idx] = clean_text_for_next_model
                    else:
                        next_iteration_doc_texts[doc_idx] = text_to_process
                
                # Unload the transformers model after processing all documents with it for this model_config
                self._unload_model(model_idx)
            
            current_doc_texts_for_processing = next_iteration_doc_texts 

        processed_documents = []
        for doc, outputs in zip(documents, outputs_by_doc):
            processed_doc = apply_tags(
                doc, [tagged_text for _, tagged_text in outputs], self.focus_tags, mode=mode, locale=locale
            )
            processed_documents.append(processed_doc)

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
