"""API clients and input mapping for LifeLine's account assistant."""
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

FIELDS = {
    "age": "Age",
    "age_missing": "AgeMissing",
    "gender": "Gender",
    "region": "Region",
    "contract_type": "ContractType",
    "monthly_charges": "MonthlyCharges",
    "signup_date": "SignupDate",
    "avg_call_minutes": "AvgCallMinutes",
    "avg_data_usage_gb": "AvgDataUsageGB",
    "avg_sms": "AvgSMS",
    "total_complaints": "TotalComplaints",
}
SCENARIO_FIELDS = {
    "total_complaints": "Total complaints",
    "monthly_charges": "Monthly charges",
    "contract_type": "Contract type",
    "avg_call_minutes": "Average call minutes",
    "avg_data_usage_gb": "Average data (GB)",
    "avg_sms": "Average SMS",
}
CONTRACTS = ("monthly", "yearly", "prepaid")
QUESTION = {
    "churn": {
        "type": "choice",
        "instructions": "Predict the customer churn label from the supplied customer attributes. Select churn for label 1 or no_churn for label 0.",
        "criteria": {
            "no_churn": "Customer does not churn (source label 0).",
            "churn": "Customer churns (source label 1).",
        },
    },
}


class ServiceError(Exception):
    """A configuration or upstream error safe to show in the dashboard."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


@dataclass(frozen=True)
class ModelService:
    name: str
    url: str = ""
    token: str = ""
    model: str = ""
    chat_path: str = "/api/chat/completions"
    models_path: str = "/api/models"

    def ready(self):
        return bool(self.url and self.token and self.model)

    def request(self, path, payload=None):
        if not self.url or not self.token:
            raise ServiceError(f"Configure the {self.name} URL and API key in Model + Settings.")
        try:
            url = urlsplit(self.url)
            if (
                url.scheme not in ("https", "http") or not url.hostname
                or url.username or url.password or url.query or url.fragment
            ):
                raise ValueError
            body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
            request = Request(
                self.url.rstrip("/") + path, data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
            )
            with build_opener(NoRedirect()).open(request, timeout=45) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise ServiceError(f"{self.name} rejected this key or its access to the endpoint.") from None
            raise ServiceError(f"{self.name} returned HTTP {exc.code}. Check its URL and model ID.") from None
        except (URLError, TimeoutError, OSError):
            raise ServiceError(f"Cannot reach {self.name} from the Streamlit server. Check the URL and network connection.") from None
        except (ValueError, TypeError):
            raise ServiceError(f"{self.name} returned invalid data or has an invalid URL.") from None

    def score(self, profile):
        if not self.model:
            raise ServiceError("Set the Unsloth decision model ID in Model + Settings.")
        output = self.request("/v1/systemone", {
            "model": self.model,
            "state": "Customer attributes:\n" + json.dumps(profile),
            "questions": QUESTION,
        })
        try:
            answer = output["answers"]["churn"]
            label = answer["choice"]
            probabilities = answer["probabilities"]
            if (
                label not in ("churn", "no_churn") or set(probabilities) != {"churn", "no_churn"}
                or any(type(n) not in (int, float) or not 0 <= n <= 1 for n in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=.01)
            ):
                raise ValueError
            return {"label": label, "score": probabilities["churn"]}
        except (KeyError, TypeError, ValueError):
            raise ServiceError("Unsloth did not return a usable churn label and score.") from None

    def model_ids(self):
        output = self.request(self.models_path)
        try:
            return [model["id"] for model in output["data"] if isinstance(model["id"], str)]
        except (KeyError, TypeError):
            raise ServiceError(f"{self.name} did not return a usable model list.") from None

    def chat(self, prompt, context):
        if not self.model:
            raise ServiceError(f"Set the {self.name} chat model ID in Model + Settings.")
        payload = {
            "model": self.model,
            "stream": False,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(context)},
            ],
        }
        if self.chat_path == "/v1/chat/completions":
            payload.update(max_tokens=192, chat_template_kwargs={"enable_thinking": False})
        output = self.request(self.chat_path, payload)
        try:
            reply = output["choices"][0]["message"]["content"].strip()
            if not reply:
                raise ValueError
            return reply
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise ServiceError(f"{self.name} returned an empty or unrecognized chat reply.") from None


def number(value, whole=False):
    if isinstance(value, bool):
        raise ValueError
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1e8 or (whole and not value.is_integer()):
        raise ValueError
    return int(value) if whole else value


def profile_for(row):
    """Translate the workbook to the exact feature names used to train Qwen."""
    profile = {}
    missing = []
    for field, column in FIELDS.items():
        try:
            value = row[column]
            if field == "age":
                flag = number(row["AgeMissing"], whole=True)
                if flag not in (0, 1):
                    raise ValueError
                value = "unknown" if flag else number(value, whole=True)
                if value != "unknown" and value > 120:
                    raise ValueError
            elif field == "age_missing":
                flag = number(value, whole=True)
                if flag not in (0, 1):
                    raise ValueError
                value = bool(flag)
            elif field == "signup_date":
                value = date.fromisoformat(str(value).split(" ")[0]).isoformat()
            elif field in ("gender", "region", "contract_type"):
                value = str(value).strip().lower()
                options = {
                    "gender": ("male", "female"),
                    "region": ("urban", "suburban", "rural"),
                    "contract_type": CONTRACTS,
                }
                if value not in options[field]:
                    raise ValueError
            else:
                value = number(value, whole=field == "total_complaints")
            profile[field] = value
        except (KeyError, TypeError, ValueError):
            missing.append(column)
    if missing:
        raise ValueError("This account needs valid values for " + ", ".join(missing) + ".")
    return profile


def parse_action(text, profile, history, speaker):
    question = text.strip().lower().rstrip(" ?!.")
    if re.fullmatch(
        r"(?:will|do you think) (?:this (?:customer|account)|the (?:customer|account)|they)(?: will)? churn(?: and why)?",
        question,
    ):
        return ("explain" if question.endswith("and why") else "score"), None

    # Keep explicit complaint comparisons usable even with a small chat model.
    complaints = re.fullmatch(
        r"(?:what if|suppose) (?:this (?:customer|account)|the (?:customer|account)|they)"
        r" (?:had|has) (?:exactly )?"
        r"(\d{1,8}|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"(?: (fewer|less|more))? complaints?(?: instead of \d+)?",
        question,
    )
    if complaints:
        amount, direction = complaints.groups()
        words = "zero one two three four five six seven eight nine ten".split()
        value = int(amount) if amount.isdecimal() else words.index(amount)
        if direction:
            value = profile["total_complaints"] + (value if direction == "more" else -value)
        if value < 0:
            return "clarify", None
        return "scenario", ("total_complaints", value)

    content = speaker.chat(
        "Route the latest question about a selected telecom account. Return only one JSON object. "
        'Use {"action":"score"} for direct prediction requests, {"action":"explain"} when asked why or for a prediction with an explanation, '
        '{"action":"scenario","field":"total_complaints","value":2} to test one change, '
        '{"action":"clarify"} if the requested scenario has multiple changes, no clear target, '
        'or uses another field. Use {"action":"chat"} for other questions. '
        "Use the original profile to calculate relative changes. Never invent an input value. "
        "A follow-up may refer to the most recent question. A scenario value must be an absolute value. "
        "Allowed scenario fields: " + json.dumps(SCENARIO_FIELDS) + ". Contract types: monthly, yearly, prepaid.",
        {"latest_question": text, "original_profile": profile, "previous_questions": history[-3:]},
    )
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        action = json.loads(content)
        if action["action"] in ("score", "explain", "chat", "clarify"):
            return action["action"], None
        if action["action"] == "scenario":
            field, value = action["field"], action["value"]
            if field not in SCENARIO_FIELDS:
                raise ValueError
            if field == "contract_type":
                if value not in CONTRACTS:
                    raise ValueError
            else:
                value = number(value, whole=field == "total_complaints")
            return "scenario", (field, value)
    except (TypeError, KeyError, ValueError):
        pass
    return "clarify", None
