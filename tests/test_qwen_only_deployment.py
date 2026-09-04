from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_models_are_git_ignored_but_available_to_model_builds() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    root_dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "models/" in gitignore
    assert "models" in root_dockerignore
    for name in ("qwen3-asr", "amphion-spec"):
        build_ignore = (ROOT / "deploy" / "docker" / f"Dockerfile.{name}.dockerignore").read_text(
            encoding="utf-8"
        )
        assert f"!models/{'qwen3-asr-1.7b' if name == 'qwen3-asr' else name}/**" in build_ignore


def test_model_images_use_local_weights_and_local_spec_plugin() -> None:
    qwen = (ROOT / "deploy" / "docker" / "Dockerfile.qwen3-asr").read_text(encoding="utf-8")
    spec = (ROOT / "deploy" / "docker" / "Dockerfile.amphion-spec").read_text(encoding="utf-8")

    assert "COPY models/qwen3-asr-1.7b" in qwen
    assert "VLLM_IMAGE=vllm/vllm-openai:v0.18.0" in qwen
    assert "COPY models/amphion-spec" in spec
    assert "COPY deploy/plugins/amphion_spec_vllm" in spec
    assert "git clone" not in spec
    assert "http://" not in spec and "https://" not in spec


def test_kubernetes_uses_bundled_images_without_model_download_resources() -> None:
    profile = ROOT / "deploy" / "k8s" / "qwen-only"
    kustomization = yaml.safe_load((profile / "kustomization.yaml").read_text(encoding="utf-8"))
    assert "model-storage.yaml" not in kustomization["resources"]
    assert all(
        generator["name"] != "amphion-spec-fetch-script"
        for generator in kustomization.get("configMapGenerator", [])
    )

    for filename in ("qwen3-asr.yaml", "amphion-spec.yaml"):
        deployment = next(
            document
            for document in yaml.safe_load_all((profile / filename).read_text(encoding="utf-8"))
            if document["kind"] == "Deployment"
        )
        assert "initContainers" not in deployment["spec"]["template"]["spec"]
        volumes = deployment["spec"]["template"]["spec"].get("volumes", [])
        assert not any("persistentVolumeClaim" in volume for volume in volumes)
