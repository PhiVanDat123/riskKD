"""Patch FastChat's llm_judge/common.py to use the openai>=1.0 Python SDK.

Upstream FastChat (v0.2.36) uses the deprecated openai<1 API:
  openai.ChatCompletion.create(...)              -> client.chat.completions.create(...)
  openai.error.OpenAIError                       -> openai.OpenAIError
  openai.error.InvalidRequestError               -> openai.BadRequestError
  response["choices"][0]["message"]["content"]   -> response.choices[0].message.content

Our environment is forced to openai>=1 because vLLM 0.21 has a hard import on openai.types.
This patch is idempotent (re-running it is a no-op once already applied).
"""
import re
import sys
from pathlib import Path


MARKER = "# === PATCH: openai>=1.x compatibility applied ==="

OLD_OPENAI_FUNC = '''def chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        openai.api_base = api_dict["api_base"]
        openai.api_key = api_dict["api_key"]
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response["choices"][0]["message"]["content"]
            break
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output'''

NEW_OPENAI_FUNC = '''def chat_completion_openai(model, conv, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        client = openai.OpenAI(api_key=api_dict["api_key"], base_url=api_dict["api_base"])
    else:
        client = openai.OpenAI()
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            messages = conv.to_openai_api_messages()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)

    return output'''


def main():
    if len(sys.argv) != 2:
        print("usage: patch_fastchat_openai_v1.py <path-to-common.py>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    src = path.read_text()
    if MARKER in src:
        print(f"[patch] already applied (marker found), no-op: {path}")
        return 0
    if OLD_OPENAI_FUNC not in src:
        print(f"[patch] ERROR: expected chat_completion_openai signature not found in {path}",
              file=sys.stderr)
        print("  This usually means upstream FastChat changed; manual review needed.",
              file=sys.stderr)
        return 1
    new_src = src.replace(OLD_OPENAI_FUNC, NEW_OPENAI_FUNC)
    # Belt-and-suspenders: rewrite the Azure variant too, but only the error-class refs
    # (we're not going to use Azure; this just keeps imports valid if someone does).
    new_src = re.sub(r"openai\.error\.OpenAIError", "openai.OpenAIError", new_src)
    new_src = re.sub(r"openai\.error\.InvalidRequestError", "openai.BadRequestError", new_src)
    new_src += f"\n\n{MARKER}\n"
    path.write_text(new_src)
    print(f"[patch] applied openai>=1.x migration to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
