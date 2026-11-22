"""Compare a live prefix with full prefill on a freshly started, single-session IQ server."""

import argparse
import json
from urllib.request import Request, urlopen


def check(base):
    with urlopen(f"{base}/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]

    def complete(prompt, limit, reuse):
        body = {
            "model": model, "prompt": prompt, "max_tokens": limit, "temperature": 0,
            "logprobs": 2, "cache_prompt": reuse, "cache_persist": False,
        }
        request = Request(f"{base}/v1/completions", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    prefix = "Sunlight is white, but the sky is blue when the weather is clear."
    prompt = prefix + " Explain why in three short sentences."
    seed = complete(prefix, 1, False)
    split = complete(prompt, 32, True)
    fresh = complete(prompt, 32, False)
    cache = split["timings"]["component_cache"]
    prefix_tokens = seed["usage"]["prompt_tokens"]
    tail = cache["prefilled_tokens"]
    if (cache["reuse"], cache["selected_tokens"], tail) != ("live", prefix_tokens, 7):
        raise AssertionError(f"Expected a live {prefix_tokens}+7 split, got {cache['reuse']} "
                             f"with {cache['selected_tokens']}+{tail} tokens")
    if fresh["usage"]["cached_tokens"] or fresh["usage"]["prompt_tokens"] != prefix_tokens + tail:
        raise AssertionError("Full-prefill control did not process the entire prompt")

    left, right = split["choices"][0], fresh["choices"][0]
    if not right["logprobs"]["tokens"]:
        raise AssertionError("Expected generated token scores")
    for field in ("text", "finish_reason", "logprobs"):
        if left[field] != right[field]:
            raise AssertionError(f"Split and full prefill differ in {field} ({prefix_tokens}+{tail} tokens)")
    print(f"Matched greedy output and top-two token scores with a {prefix_tokens}+{tail} token split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:8080")
    check(parser.parse_args().base.rstrip("/"))
