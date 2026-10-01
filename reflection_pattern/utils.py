from dotenv import load_dotenv
load_dotenv()

import base64
import mimetypes
import os
import re

import pandas as pd


def _in_notebook() -> bool:
    """True when running inside Jupyter/IPython (where HTML can be rendered)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def print_html(content, title: str | None = None, is_image: bool = False) -> None:
    """Show a DataFrame, text/code, or an image file nicely.

    content   - a DataFrame/Series, a string (text or code), or, when
                is_image=True, the path to an image file such as "chart_v1.png".
    title     - optional heading shown above the content.
    is_image  - set True when `content` is an image file path.

    In a Jupyter notebook the content is rendered (HTML table, formatted code
    block, or the actual picture). In a normal terminal / .py script, which
    can't render HTML or images, it prints plain text instead (for an image,
    its file path).
    """
    import html as html_lib

    in_nb = _in_notebook()

    # ---------- Image ----------
    if is_image:
        path = str(content)
        if not os.path.exists(path):
            print(f"{title + ': ' if title else ''}image not found: {path}")
            return
        if in_nb:
            from IPython.display import HTML, Image, display
            if title:
                display(HTML(f"<h4>{html_lib.escape(title)}</h4>"))
            display(Image(filename=path))
        else:
            print(f"{title + ': ' if title else ''}image saved at {os.path.abspath(path)}")
        return

    # ---------- DataFrame / Series ----------
    if isinstance(content, (pd.DataFrame, pd.Series)):
        body_html = content.to_html()
        body_text = content.to_string()
    # ---------- Text / code ----------
    else:
        text_value = str(content)
        # Escape it and wrap in <pre> so tags like <execute_python> stay visible
        body_html = f"<pre style='white-space:pre-wrap'>{html_lib.escape(text_value)}</pre>"
        body_text = text_value

    if title:
        body_html = f"<h4>{html_lib.escape(title)}</h4>" + body_html
        body_text = f"{title}\n{body_text}"

    if in_nb:
        from IPython.display import HTML, display
        display(HTML(body_html))
    else:
        print(body_text)


def get_response(model: str, prompt: str, max_tokens: int = 4096) -> str:
    """Send a text prompt to an LLM and return its reply as one string.

    Picks the provider from the model name:
    - names containing "claude" or "anthropic" -> Anthropic (needs ANTHROPIC_API_KEY)
    - everything else                          -> OpenAI    (needs OPENAI_API_KEY)
    """
    lower = model.lower()

    if "claude" in lower or "anthropic" in lower:
        # Imported here so the file still works if you only use one provider
        from anthropic import Anthropic

        client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # The reply can come back in several blocks; join all the text ones
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )

    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""


def encode_image_b64(path: str) -> tuple[str, str]:
    """Read an image file and return (media_type, base64_string).

    Example: ("image/png", "iVBORw0KGgo...")
    APIs can't take a file path, so the image is sent as base64 text.
    Raises FileNotFoundError if the file doesn't exist.
    """
    media_type, _ = mimetypes.guess_type(path)
    if media_type is None or not media_type.startswith("image/"):
        media_type = "image/png"  # sensible default for matplotlib output

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return media_type, b64


def image_anthropic_call(
    model_name: str, prompt: str, media_type: str, b64: str, max_tokens: int = 4096
) -> str:
    """Send an image + text prompt to a Claude model; return the reply as one string."""
    from anthropic import Anthropic

    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    message = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        system="You are a careful data visualization expert. Follow the requested output format exactly.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    # The reply can come back in several blocks; join all the text ones
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )


def image_openai_call(model_name: str, prompt: str, media_type: str, b64: str) -> str:
    """Send an image + text prompt to an OpenAI vision model; return the reply as one string.

    Works with vision-capable models such as gpt-4o, gpt-4o-mini and o4-mini.
    No temperature / max_tokens is set, because reasoning models (o-series)
    reject some of those parameters.
    """
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return completion.choices[0].message.content or ""


def ensure_execute_python_tags(code: str) -> str:
    """Return the code wrapped in <execute_python> ... </execute_python> tags.

    Safe to call on anything:
    - already-tagged text is unwrapped first, so tags are never doubled
    - stray markdown fences (```python ... ```) are removed, since they would
      cause a SyntaxError when the code is run with exec()
    """
    code = (code or "").strip()

    tagged = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code)
    if tagged:
        code = tagged.group(1).strip()

    code = re.sub(r"^```(?:python|py)?[ \t]*\n", "", code)
    code = re.sub(r"\n?```[ \t]*$", "", code).strip()

    return f"<execute_python>\n{code}\n</execute_python>"


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the coffee sales CSV and return a DataFrame with the expected types."""
    df = pd.read_csv(csv_path)

    # Convert 'date' from text to datetime64 so .dt.year, .dt.month, etc. work
    df["date"] = pd.to_datetime(df["date"])

    # 'time' stays a plain string (HH:MM); do not combine it with 'date'
    df["time"] = df["time"].astype(str)

    # Make sure the numeric columns have the right types
    df["price"] = df["price"].astype(float)

    # Recompute these from 'date' so they always match it
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["quarter"] = df["date"].dt.quarter.astype(int)

    return df