#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
import os
from langchain_core.output_parsers import JsonOutputParser



QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    PROMPT = """
        You are reading receipts.
    
        Extract THREE numbers and return them as JSON.
    
        Return ONLY a JSON object with exactly these three keys:
        {{
        "amount_paid_after_rounding": <number>,
        "subtotal_after_discounts_before_rounding": <number>,
        "discount_list": [<number>, <number>, ...]
        }}
    
        Definitions:
    
        1. amount_paid_after_rounding:
        - It is the amount of money you actually pay after the rounding
        - You will find the amount right below the rounding and above the change
        - Its label is often OCTOPUS or VISA, but could be MASTERCARD, EPS, ALIPAY, CASH, TOTAL, or another payment name.
        - The label appears on the line directly below ROUNDING, NOT in the product list.
        - Example: "ROUNDING -$0.05" then "OCTOPUS $123.45" -> 123.45
    
        2. subtotal_after_discounts_before_rounding:
        - It is the amount of money that all the positive and negatives amounts add up to from the list of bought items
        - It is the amount of money that is listed before rounding and after the discount and if there is no discount, the list of bought items
        - Do NOT include the rounding line in this sum.
        - Its label is SUBTOTAL or 小計.
        - Example: "20 小計 $123.45" -> 123.45
    
        3. discount_list:
        - A list of every discount line in the item list (above the SUBTOTAL line).
        - Report each as a POSITIVE number.
        - Discounts appear as negative amounts in the item list (e.g. -$12.40), or lines with "Save", "% OFF", or 包裝變形.
        - Use the AMOUNT from the right-hand column, not the number in the label (e.g. "Save $5  -$6.00" -> 6.00, not 5).
        - Exclude the ROUNDING line, the CHANGE line, and anything below the SUBTOTAL line.
        - Do NOT sum them. Just list them.
        - Be careful: the discount lines are scattered throughout the receipt. Scan every line.
        - If there are no discounts, return [0].
        - Example: [1.00, 2.00, 3.00]
    
        Return ONLY the JSON. No explanation. No markdown fences.
        """
    
    ### YOUR CODE HERE
    llm = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.1,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT),
        ("human", [
            {"type": "text", "text": "Extract the three numbers from this receipt. Return JSON only."},
            {"type": "image_url", "image_url": "{image}"},
        ]),
    ])
    
    return prompt | llm



def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    parser = JsonOutputParser()
    receipt_list = []

    for img in images:
        data_url = image_data_url(img)
        result = chain.invoke({"image": data_url})
        text = response_text(result)
        parsed = parser.parse(text)
        print(f"{img.name}: {parsed}")
        receipt_list.append(parsed)

    query1 = sum(float(r["amount_paid_after_rounding"]) for r in receipt_list)
    query2 = sum(
        float(r["subtotal_after_discounts_before_rounding"]) + sum(r.get("discount_list", [0]))
        for r in receipt_list
    )

    return {
        QUERY_1: f"HK${query1:.2f}",
        QUERY_2: f"HK${query2:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
