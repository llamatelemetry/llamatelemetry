import types
from pathlib import Path

from llamatelemetry.server import ServerManager


class _SplitMode:
    value = "layer"


class _MultiGPUConfig:
    def __init__(self):
        self.n_gpu_layers = 42
        self.main_gpu = 1
        self.split_mode = _SplitMode()
        self.tensor_split = [0.5, 0.5]
        self.use_mmap = True
        self.use_mlock = False
        self.flash_attention = True
        self.no_kv_offload = False
        self.ctx_size = 0
        self.batch_size = 0
        self.ubatch_size = 0


def test_multi_gpu_config_maps_split_args(tmp_path, monkeypatch):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"dummy")

    manager = ServerManager(server_url="http://127.0.0.1:8090")

    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            self.stderr = types.SimpleNamespace(read=lambda: b"")

        def poll(self):
            return None

    call = {"n": 0}

    def fake_check(*args, **kwargs):
        call["n"] += 1
        return call["n"] > 1

    monkeypatch.setattr(manager, "find_llama_server", lambda: Path("/bin/echo"))
    monkeypatch.setattr(manager, "check_server_health", fake_check)
    monkeypatch.setattr("llamatelemetry.server.subprocess.Popen", _FakePopen)

    cfg = _MultiGPUConfig()
    ok = manager.start_server(
        model_path=str(model_path),
        skip_gpu_check=True,
        multi_gpu_config=cfg,
        timeout=1,
        verbose=False,
        enable_slots=True,
    )

    assert ok is True
    cmd = captured["cmd"]
    assert "--split-mode" in cmd
    split_index = cmd.index("--split-mode")
    assert cmd[split_index + 1] == "layer"
    assert "--tensor-split" in cmd
    tensor_index = cmd.index("--tensor-split")
    assert cmd[tensor_index + 1] == "0.5,0.5"
