import types


def test_multi_gpu_config_not_forwarded_to_cli(monkeypatch, tmp_path):
    from llamatelemetry.server import ServerManager

    manager = ServerManager(server_url="http://127.0.0.1:8090")
    model_path = tmp_path / "model.gguf"
    model_path.write_text("dummy")

    # Avoid filesystem/network dependencies
    monkeypatch.setattr(manager, "find_llama_server", lambda: "/bin/echo")
    monkeypatch.setattr(manager, "check_server_health", lambda timeout=1.0: True)

    captured = {}

    class DummyProc:
        stderr = None

        def poll(self):
            return None

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return DummyProc()

    import llamatelemetry.server as server_module
    monkeypatch.setattr(server_module.subprocess, "Popen", fake_popen)

    # Pass a sentinel multi_gpu_config to ensure it never becomes a CLI flag
    manager.start_server(
        model_path=str(model_path),
        skip_gpu_check=True,
        multi_gpu_config=types.SimpleNamespace(),
    )

    cmd = captured.get("cmd", [])
    assert "--multi-gpu-config" not in cmd
