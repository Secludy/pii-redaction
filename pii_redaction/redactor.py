import torch
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from collections import defaultdict
import difflib
from enum import Enum
from .faker_utils import FakePIIGenerator


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
    def __init__(self, device=None):
        """
        Initialize the PIIRedactor with models for PII detection.

        Args:
            device (str): Device to use for inference (e.g., 'cuda', 'cpu')
        """

        self.model_paths = ["OpenPipe/Pii-Redact-Name", "OpenPipe/Pii-Redact-General"]
        self.focus_tags = [["person_name", "organization_name"]] + [None] * (
            len(self.model_paths) - 1
        )

        self.device = device

        self.models = [None] * len(self.model_paths)
        self.tokenizers = [None] * len(self.model_paths)

    def _initialize_model(self, index):
        """Initialize a specific model if it hasn't been already."""
        if self.models[index] is None:
            self.models[index] = AutoModelForCausalLM.from_pretrained(
                self.model_paths[index]
            )
            self.tokenizers[index] = AutoTokenizer.from_pretrained(
                self.model_paths[index], padding_side="left"
            )
            self.tokenizers[index].padding_side = "left"

            if self.device:
                self.models[index] = self.models[index].to(self.device)
            elif torch.cuda.is_available():
                self.models[index] = self.models[index].to("cuda")

    def _model_call(self, text, model_index):
        """
        Process text through a specific model to identify PII entities.

        Args:
            text (str): The text to process
            model_index (int): Index of the model to use

        Returns:
            str: The processed text with PII tags
        """
        self._initialize_model(model_index)

        model = self.models[model_index]
        tokenizer = self.tokenizers[model_index]

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        messages = [{"role": "user", "content": text}]

        encoded_input = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        input_ids = encoded_input["input_ids"].to(model.device)
        attention_mask = encoded_input["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = encoded_input["input_ids"].size(1)
        generated_ids = outputs[0][input_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

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
        processed_documents = []

        for doc in documents:
            model_outputs = []

            for idx in range(len(self.model_paths)):
                model_output = self._model_call(doc, idx)
                model_outputs.append(model_output)

            processed_doc = apply_tags(
                doc, model_outputs, self.focus_tags, mode=mode, locale=locale
            )
            processed_documents.append(processed_doc)

        return processed_documents


def tag_pii_in_documents(
    documents, device=None, mode=PIIHandlingMode.TAG, locale="en_US"
):
    """
    Convenience function to process a list of documents through a PII tagging model.

    Args:
        documents (list): List of text documents to process.
        device (str): Device to use for processing (e.g., 'cuda', 'cpu').
        mode (PIIHandlingMode): How to handle identified PII:
            - TAG: Keep PII with XML tags
            - REDACT: Replace PII with empty tags
            - REPLACE: Replace PII with fake data
        locale (str): Locale for generating fake data (only used if mode=REPLACE)

    Returns:
        list: List of documents with PII handled according to the specified mode.
    """
    redactor = PIIRedactor(device=device)
    return redactor.tag_pii_in_documents(documents, mode=mode, locale=locale)


def clean_dataset(
    input_filename,
    output_filename,
    device=None,
    mode=PIIHandlingMode.TAG,
    locale="en_US",
):
    """
    Reads a JSONL dataset and processes the 'content' field in each message.
    Processes JSON objects, updates them with the processed messages,
    and writes them immediately to the output file. This allows progress to be saved incrementally.

    Args:
        input_filename (str): Path to the input JSONL file.
        output_filename (str): Path to the output JSONL file.
        device (str): Device to use for processing (e.g., 'cuda', 'cpu').
        mode (PIIHandlingMode): How to handle identified PII:
            - TAG: Keep PII with XML tags
            - REDACT: Replace PII with empty tags
            - REPLACE: Replace PII with fake data
        locale (str): Locale for generating fake data (only used if mode=REPLACE)
    """
    redactor = PIIRedactor(device=device)

    with open(input_filename, "r") as f:
        num_lines = sum(1 for line in f)

    with open(input_filename, "r") as fin, open(output_filename, "w") as fout:
        for line in tqdm(fin, total=num_lines):
            json_obj = json.loads(line.strip())

            process_and_write_batch(
                [json_obj], fout, redactor, mode=mode, locale=locale
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
