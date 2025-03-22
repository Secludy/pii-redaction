# PII Redaction

A Python package for redacting Personally Identifiable Information (PII) from text using Large Language Models.

## Installation

```bash
pip install pii-redaction
```

Or install from source:

```bash
git clone https://github.com/yourusername/pii-redaction.git
cd pii-redaction
pip install -e .
```

## Usage

### Command Line Interface

The package provides a command-line tool `pii-redact` with the following commands:

#### Process a JSONL dataset

For handling PII in JSONL files that contain messages (like conversation history):

```bash
pii-redact process-jsonl input.jsonl output.jsonl
```

Options:
- `--device`: Device to use for processing (e.g., cuda, cpu)
- PII handling modes (mutually exclusive):
  - `--tag`: Keep PII content between XML tags (default) `<PII:type>content</PII:type>`
  - `--redact`: Replace PII with just an empty tag `<PII:type/>`
  - `--replace`: Replace PII with fake data `fake_data`
- `--locale`: Locale for generating fake data (default: en_US, only used with --replace)

#### Process text files

For handling PII in plain text files (one document per line):

```bash
pii-redact process-text input.txt output.txt
```

Options:
- `--device`: Device to use for processing (e.g., cuda, cpu)
- PII handling modes (mutually exclusive):
  - `--tag`: Keep PII content between XML tags (default) `<PII:type>content</PII:type>`
  - `--redact`: Replace PII with just an empty tag `<PII:type/>`
  - `--replace`: Replace PII with fake data `fake_data`
- `--locale`: Locale for generating fake data (default: en_US, only used with --replace)

#### Examples

Tag PII in text documents (default mode):
```bash
pii-redact process-text emails.txt tagged_emails.txt
```

Redact PII completely:
```bash
pii-redact process-text emails.txt redacted_emails.txt --redact
```

Replace PII with fake data:
```bash
pii-redact process-text emails.txt anonymized_emails.txt --replace
```

Use a specific locale for fake data:
```bash
pii-redact process-text emails.txt anonymized_emails.txt --replace --locale=fr_FR
```

Process a JSONL dataset and redact PII:
```bash
pii-redact process-jsonl conversations.jsonl redacted_conversations.jsonl --redact
```

### Python API

```python
from pii_redaction import tag_pii_in_documents, clean_dataset, PIIHandlingMode

# Process text documents
documents = [
    "My name is John Doe and my email is john.doe@example.com",
    "Call me at 555-123-4567 and ask for my SSN: 123-45-6789"
]

# Tag PII (default mode)
tagged_documents = tag_pii_in_documents(documents, mode=PIIHandlingMode.TAG)

# Redact PII completely
redacted_documents = tag_pii_in_documents(documents, mode=PIIHandlingMode.REDACT)

# Replace PII with fake data
anonymized_documents = tag_pii_in_documents(
    documents, 
    mode=PIIHandlingMode.REPLACE,
    locale="en_US"
)

# Process a JSONL dataset
# Tag PII (default mode)
clean_dataset('input.jsonl', 'output.jsonl', mode=PIIHandlingMode.TAG)

# Redact PII in a JSONL dataset
clean_dataset('input.jsonl', 'redacted.jsonl', mode=PIIHandlingMode.REDACT)

# Replace PII with fake data in a JSONL dataset
clean_dataset(
    'input.jsonl', 
    'anonymized.jsonl', 
    mode=PIIHandlingMode.REPLACE,
    locale="en_US"
)
```

#### Key Features

**Multiple PII handling options**:
   - **Tag PII**: Identify and keep PII with XML tags like `<PII:email_address>john.doe@example.com</PII:email_address>`
   - **Redact PII**: Replace PII with just an empty tag like `<PII:email_address/>`
   - **Replace PII**: Replace identified PII with realistic fake data like `<PII:email_address>jane.smith@example.org</PII:email_address>`

**Customizable**: Choose from different locales for generating culturally appropriate fake data
**Consistent replacement**: When replacing PII with fake data, maintains consistency (same PII values are replaced with the same fake values)

## Supported PII Categories

The model can identify and tag the following PII categories:

- age: a person's age
- credit_card_info: a credit card number, expiration date, CCV, etc.
- nationality: a country when used to reference place of birth, residence, or citizenship
- date: a specific calendar date
- date_of_birth: a specific calendar date representing birth
- domain_name: a domain on the internet
- email_address: an email ID
- demographic_group: Anything that identifies race or ethnicity
- gender: a gender identifier
- personal_id: Any ID string like a national ID, subscriber number, etc.
- other_id: Any ID not associated with a person like an organization ID, database ID, etc.
- banking_number: a number associated with a bank account
- medical_condition: A diagnosis, treatment code or other information identifying a medical condition
- organization_name: name of an organization
- person_name: name of a person
- phone_number: a telephone number
- street_address: a physical address
- password: a secure string used for authentication
- secure_credential: any secure credential like an API key, private key, 2FA token
- religious_affiliation: anything that identifies religious affiliation

## License

MIT