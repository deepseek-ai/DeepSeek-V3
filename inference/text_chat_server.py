"""
import os
import json
import glob
import threading
from flask import Flask, request, Response, render_template_string
import torch

from transformers import AutoTokenizer

from model import Transformer, ModelArgs
from generate import generate

app = Flask(__name__)

_lock = threading.Lock()
_backend = {"loaded": False, "model": None, "tokenizer": None, "args": None, "device": "cpu"}

TEXT_CHAT_HTML = """<!doctype html>
<html>
  <head><meta charset="utf-8"><title>DeepSeek Text Chat (no-JS)</title></head>
  <body>
    <h2>DeepSeek — text-only chat</h2>
    <form method="post" action="/text-chat">
      <label for="message">Message:</label><br>
      <textarea id="message" name="message" rows="4" cols="60">{{message}}</textarea><br>
      <input type="submit" value="Send">
    </form>
    {% if reply %}
    <h3>Reply</h3>
    <pre>{{reply}}</pre>
    {% endif %}
    <p>If this server is not configured with local model weights, use the /api/text-chat endpoint against your deployment.</p>
  </body>
</html>
"""


def _find_checkpoint_file(ckpt_dir):
    patterns = ["model*.safetensors", "model*-mp*.safetensors", "*.safetensors"]
    for p in patterns:
        matches = glob.glob(os.path.join(ckpt_dir, p))
        if matches:
            return matches[0]
    return None


def _load_backend():
    with _lock:
        if _backend["loaded"]:
            return
        ckpt_path = os.getenv("DEEPSEEK_CKPT_PATH")
        config_path = os.getenv("DEEPSEEK_CONFIG_PATH")
        if not ckpt_path or not config_path:
            _backend["loaded"] = False
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            with open(config_path, "r") as f:
                args = ModelArgs(**json.load(f))
            model = Transformer(args)
            tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
            ckpt_file = _find_checkpoint_file(ckpt_path)
            if ckpt_file:
                try:
                    from safetensors.torch import load_model as _load_model
                    _load_model(model, ckpt_file)
                except Exception:
                    # best-effort load; if it fails we keep backend unloaded
                    _backend["loaded"] = False
                    return
            else:
                _backend["loaded"] = False
                return
            _backend.update({"loaded": True, "model": model, "tokenizer": tokenizer, "args": args, "device": device})
        except Exception:
            _backend["loaded"] = False


def _ensure_backend():
    if not _backend["loaded"]:
        _load_backend()


def _generate_reply(message, max_new_tokens=200, temperature=1.0):
    _ensure_backend()
    if not _backend["loaded"] or _backend["model"] is None or _backend["tokenizer"] is None:
        raise RuntimeError("Local model backend not configured or failed to load. Set DEEPSEEK_CKPT_PATH and DEEPSEEK_CONFIG_PATH environment variables with a valid local checkpoint and config if you want local generation.")
    tokenizer = _backend["tokenizer"]
    model = _backend["model"]
    try:
        prompt_tokens = tokenizer.apply_chat_template([{"role": "user", "content": message}], add_generation_prompt=True)
    except Exception:
        prompt_tokens = tokenizer.encode(message)
    completion_tokens = generate(model, [prompt_tokens], max_new_tokens, getattr(tokenizer, "eos_token_id", -1), temperature)
    try:
        completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
    except Exception:
        completion = " ".join(map(str, completion_tokens[0]))
    return completion


@app.route("/api/text-chat", methods=["POST"])
def api_text_chat():
    if request.is_json:
        payload = request.get_json()
        message = payload.get("message") if isinstance(payload, dict) else None
    else:
        message = request.form.get("message") or request.values.get("message")
    if not message:
        return Response("Missing 'message' parameter", status=400, mimetype="text/plain; charset=utf-8")
    try:
        reply = _generate_reply(message)
    except RuntimeError as e:
        return Response(str(e), status=503, mimetype="text/plain; charset=utf-8")
    return Response(reply, status=200, mimetype="text/plain; charset=utf-8")


@app.route("/text-chat", methods=["GET", "POST"])
def text_chat_page():
    if request.method == "GET":
        return render_template_string(TEXT_CHAT_HTML, message="", reply=None)
    message = request.form.get("message", "")
    if not message:
        return render_template_string(TEXT_CHAT_HTML, message=message, reply="No message provided")
    try:
        reply = _generate_reply(message)
    except RuntimeError as e:
        return render_template_string(TEXT_CHAT_HTML, message=message, reply=str(e))
    return render_template_string(TEXT_CHAT_HTML, message=message, reply=reply)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
"""