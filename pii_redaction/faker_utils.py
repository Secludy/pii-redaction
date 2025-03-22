"""
Utilities for generating fake PII data to replace redacted content using the Faker library.
"""

from faker import Faker
from typing import Dict


class FakePIIGenerator:
    """
    Generates fake data for different PII categories using Faker.
    Maintains consistency by remembering previously generated values.
    """

    def __init__(self, locale: str = "en_US"):
        """
        Initialize the fake data generator.

        Args:
            locale: Locale for generating fake data
        """
        self.faker = Faker(locale)

        self.memory: Dict[str, Dict[str, str]] = {}
        for category in self.get_supported_categories():
            self.memory[category] = {}

    @staticmethod
    def get_supported_categories():
        """Returns a list of supported PII categories."""
        return [
            "age",
            "credit_card_info",
            "nationality",
            "date",
            "date_of_birth",
            "domain_name",
            "email_address",
            "demographic_group",
            "gender",
            "personal_id",
            "other_id",
            "banking_number",
            "medical_condition",
            "organization_name",
            "person_name",
            "phone_number",
            "street_address",
            "password",
            "secure_credential",
            "religious_affiliation",
        ]

    def get_fake_value(self, category: str, original_value: str) -> str:
        """
        Get a fake value for a given category and original value.
        If the original value has been seen before, returns the same fake value.

        Args:
            category: PII category
            original_value: Original PII value to replace

        Returns:
            Fake value to use as replacement
        """
        # Check if we've already generated a fake value for this original value
        if original_value in self.memory[category]:
            return self.memory[category][original_value]

        # Generate a new fake value
        method_name = f"_generate_{category.lower()}"
        if hasattr(self, method_name):
            generator = getattr(self, method_name)
            fake_value = generator(original_value)
        else:
            # Fallback for unsupported categories
            fake_value = self._generate_generic(original_value)

        # Remember this mapping for future use
        self.memory[category][original_value] = fake_value
        return fake_value

    def _generate_age(self, original: str) -> str:
        """Generate a fake age."""
        return str(self.faker.random_int(min=18, max=90))

    def _generate_credit_card_info(self, original: str) -> str:
        """Generate a fake credit card number."""
        return self.faker.credit_card_number(card_type=None)

    def _generate_nationality(self, original: str) -> str:
        """Generate a fake nationality."""
        return self.faker.country()

    def _generate_date(self, original: str) -> str:
        """Generate a fake date."""
        return self.faker.date()

    def _generate_date_of_birth(self, original: str) -> str:
        """Generate a fake date of birth."""
        return self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime(
            "%Y-%m-%d"
        )

    def _generate_domain_name(self, original: str) -> str:
        """Generate a fake domain name."""
        return self.faker.domain_name()

    def _generate_email_address(self, original: str) -> str:
        """Generate a fake email address."""
        return self.faker.email()

    def _generate_demographic_group(self, original: str) -> str:
        """Generate a fake demographic group."""
        return self.faker.word()

    def _generate_gender(self, original: str) -> str:
        """Generate a fake gender."""
        return self.faker.random_element(
            elements=("Male", "Female", "Non-binary", "Prefer not to say")
        )

    def _generate_personal_id(self, original: str) -> str:
        """Generate a fake personal ID."""
        return self.faker.ssn()

    def _generate_other_id(self, original: str) -> str:
        """Generate a fake organization ID."""
        return f"ID-{self.faker.uuid4()[:8]}"

    def _generate_banking_number(self, original: str) -> str:
        """Generate a fake banking number."""
        return self.faker.bban()

    def _generate_medical_condition(self, original: str) -> str:
        """Generate a fake medical condition."""
        conditions = [
            "Common Cold",
            "Seasonal Allergies",
            "Migraine",
            "Minor Sprain",
            "General Checkup",
        ]
        return self.faker.random_element(elements=conditions)

    def _generate_organization_name(self, original: str) -> str:
        """Generate a fake organization name."""
        return self.faker.company()

    def _generate_person_name(self, original: str) -> str:
        """Generate a fake person name."""
        return self.faker.name()

    def _generate_phone_number(self, original: str) -> str:
        """Generate a fake phone number."""
        return self.faker.phone_number()

    def _generate_street_address(self, original: str) -> str:
        """Generate a fake street address."""
        return self.faker.street_address()

    def _generate_password(self, original: str) -> str:
        """Generate a fake password."""
        return self.faker.password(
            length=12, special_chars=True, digits=True, upper_case=True, lower_case=True
        )

    def _generate_secure_credential(self, original: str) -> str:
        """Generate a fake secure credential."""
        return f"FAKE_API_KEY_{self.faker.uuid4()}"

    def _generate_religious_affiliation(self, original: str) -> str:
        """Generate a fake religious affiliation."""
        affiliations = ["Religion A", "Faith B", "Belief System C", "Spiritual Group D"]
        return self.faker.random_element(elements=affiliations)

    def _generate_generic(self, original: str) -> str:
        """Fallback generator for unsupported categories."""
        return f"REPLACED_DATA_{self.faker.uuid4()[:8]}"
