"""llmx image --auth api: GPT Image 2.6 request shapes, with the OpenAI client mocked."""
import base64
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from llmx import image
from llmx.cli import image_cmd


@pytest.fixture
def sdk(monkeypatch):
    calls = []

    def respond(endpoint):
        def call(**kwargs):
            markup = kwargs.get("markup")
            calls.append({"endpoint": endpoint, "markup_open": markup and not markup[1].closed, **kwargs})
            return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"bytes").decode())], usage=None)
        return call

    client = SimpleNamespace(images=SimpleNamespace(generate=respond("generate"), edit=respond("edit")))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(image, "check_openai_api_key", lambda: "key")
    monkeypatch.setattr("llmx.usage_log.log_usage", lambda **kwargs: None)
    return calls


@pytest.mark.parametrize("alias,model", [
    ("sunburst", "gpt-image-2.6-sunburst"),
    ("flare", "gpt-image-2.6-flare"),
    ("gpt-image-2.6", "gpt-image-2.6-sunburst"),
    ("gpt-image-2.5", "gpt-image-2.5-sunburst"),
    ("gpt-image-2.5-flare", "gpt-image-2.5-flare"),
])
def test_alias_resolution(sdk, tmp_path, alias, model):
    image.generate_openai_image("p", str(tmp_path / "o.png"), model=alias)
    assert sdk[0]["model"] == model


def test_default_is_sunburst_2_6_and_omits_new_fields(sdk, tmp_path):
    image.generate_openai_image("p", str(tmp_path / "o.png"))
    call = sdk[0]
    assert call["model"] == "gpt-image-2.6-sunburst"
    assert not {"complexity", "frame_count", "frame_duration_ms", "markup"} & call.keys()


def test_gif_request(sdk, tmp_path):
    out = tmp_path / "o.gif"
    paths = image.generate_openai_image(
        "p", str(out), model="flare", output_format="gif", frame_count=6, frame_duration_ms=150,
        complexity="low", size="1024x1024",
    )
    call = sdk[0]
    assert (call["endpoint"], call["output_format"]) == ("generate", "gif")
    assert (call["frame_count"], call["frame_duration_ms"], call["complexity"]) == (6, 150, "low")
    assert paths == [out] and out.read_bytes() == b"bytes"


def test_svg_default_path_gets_svg_suffix(sdk, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = image.generate_openai_image("logo", output_format="svg")
    assert sdk[0]["output_format"] == "svg"
    assert sdk[0]["svg"] == {"model": "gpt-6.1-sol"}
    assert paths[0].suffix == ".svg"


def test_svg_author_override(sdk, tmp_path):
    image.generate_openai_image(
        "logo", str(tmp_path / "o.svg"), output_format="svg", svg_model="gpt-6-astra", svg_effort="low",
    )
    assert sdk[0]["svg"] == {"model": "gpt-6-astra", "reasoning": {"effort": "low"}}


def test_markup_edit_sends_open_png_tuple(sdk, tmp_path):
    base, layer = tmp_path / "base.png", tmp_path / "layer.png"
    base.write_bytes(b"a")
    layer.write_bytes(b"b")
    image.generate_openai_image("p", str(tmp_path / "o.png"), input_images=[str(base)], markup=str(layer), complexity="low")
    call = sdk[0]
    assert call["endpoint"] == "edit" and call["complexity"] == "low"
    assert call["markup_open"] is True
    assert call["markup"][0] == "layer.png" and call["markup"][2] == "image/png"
    assert call["markup"][1].closed


@pytest.mark.parametrize("kwargs,message", [
    ({"output_format": "svg", "n": 2}, "n=1"),
    ({"output_format": "gif", "input_images": ["x.png"]}, "generation-only"),
    ({"output_format": "svg", "input_images": ["x.png"]}, "generation-only"),
    ({"output_format": "gif", "output_path": "x.png"}, r"\.gif path"),
    ({"frame_count": 5}, "only to gif"),
    ({"frame_duration_ms": 100}, "only to gif"),
    ({"svg_model": "gpt-6-astra"}, "only to svg"),
    ({"output_format": "gif", "svg_effort": "low"}, "only to svg"),
    ({"output_format": "gif", "frame_count": 1}, "between 2 and 16"),
    ({"output_format": "gif", "frame_duration_ms": 10}, "between 20 and 2000"),
    ({"complexity": "extreme"}, "complexity"),
    ({"markup": "x.png"}, "first input image"),
    ({"size": "1000x1000"}, "multiples of 16"),
    ({"size": "big"}, "multiples of 16"),
])
def test_invalid_requests_rejected_before_sdk(sdk, kwargs, message):
    with pytest.raises(ValueError, match=message):
        image.generate_openai_image("p", **kwargs)
    assert not sdk


@pytest.mark.parametrize("size", ["auto", "1024x1024", "2048x3072", "3840x2160"])
def test_flexible_sizes_pass(sdk, tmp_path, size):
    image.generate_openai_image("p", str(tmp_path / "o.png"), size=size)
    assert sdk[0]["size"] == size


@pytest.mark.parametrize("args", [
    ["--complexity", "low"],
    ["--format", "gif"],
    ["--format", "svg"],
    ["--provider", "google", "--complexity", "low"],
])
def test_api_only_flags_fail_loud_off_the_api_route(monkeypatch, args):
    def must_not_run(*a, **k):
        raise AssertionError("no backend may run")

    monkeypatch.setattr(image, "generate_codex_image", must_not_run)
    monkeypatch.setattr(image, "generate_image", must_not_run)
    result = CliRunner().invoke(image_cmd, ["p", *args])
    assert result.exit_code != 0
    assert "--auth api" in result.output
