import json

from colab_cli.ws_kernel import WSKernelClient


class ImmediateReplyWebSocket:
    """Replies inside send(), exposing registration-after-send races."""

    def __init__(self, client):
        self.client = client

    def send(self, raw):
        request = json.loads(raw)
        parent_id = request["header"]["msg_id"]
        messages = [
            {
                "channel": "iopub",
                "header": {"msg_type": "stream"},
                "parent_header": {"msg_id": parent_id},
                "content": {"name": "stdout", "text": "fast\n"},
            },
            {
                "channel": "shell",
                "header": {"msg_type": "execute_reply"},
                "parent_header": {"msg_id": parent_id},
                "content": {"status": "ok"},
            },
            {
                "channel": "iopub",
                "header": {"msg_type": "status"},
                "parent_header": {"msg_id": parent_id},
                "content": {"execution_state": "idle"},
            },
        ]
        for message in messages:
            self.client._dispatch(json.dumps(message))


def test_execute_registers_queues_before_send():
    client = WSKernelClient("https://example.invalid", "token")
    client.session_id = "session"
    client._ws = ImmediateReplyWebSocket(client)

    result = client.execute("1 + 1", timeout=0.1)
    assert result["status"] == "ok"
    assert result["outputs"][0]["text"] == "fast\n"


def test_session_shim_tracks_session_created_during_start():
    client = WSKernelClient("https://example.invalid", "token")
    assert client._manager.client.session.session is None
    client.session_id = "created-later"
    assert client._manager.client.session.session == "created-later"


def test_start_removes_handshake_timeout_from_connected_socket(monkeypatch):
    class Socket:
        def __init__(self):
            self.timeout = "unset"

        def settimeout(self, value):
            self.timeout = value

        def recv(self):
            return ""

    socket = Socket()
    monkeypatch.setattr(
        "colab_cli.ws_kernel.websocket.create_connection",
        lambda *args, **kwargs: socket,
    )
    monkeypatch.setattr(
        "colab_cli.ws_kernel.threading.Thread.start", lambda self: None
    )

    client = WSKernelClient(
        "https://example.invalid", "token", kernel_id="kernel"
    )
    client.start(timeout=12)
    assert socket.timeout is None


def test_activity_resets_inactivity_timeout(monkeypatch):
    client = WSKernelClient("https://example.invalid", "token")
    client.session_id = "session"

    values = [0.0, 0.0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 4.5]
    monotonic_times = iter(values)
    wall_times = iter(values)
    monkeypatch.setattr(
        "colab_cli.ws_kernel.time.monotonic", lambda: next(monotonic_times)
    )
    # This also makes the test fail against the old wall-clock implementation.
    monkeypatch.setattr("colab_cli.ws_kernel.time.time", lambda: next(wall_times))

    class ActiveWebSocket:
        def send(self, raw):
            parent_id = json.loads(raw)["header"]["msg_id"]
            for text in ("one", "two"):
                client._dispatch(
                    json.dumps(
                        {
                            "channel": "iopub",
                            "header": {"msg_type": "stream"},
                            "parent_header": {"msg_id": parent_id},
                            "content": {"name": "stdout", "text": text},
                        }
                    )
                )
            client._dispatch(
                json.dumps(
                    {
                        "channel": "shell",
                        "header": {"msg_type": "execute_reply"},
                        "parent_header": {"msg_id": parent_id},
                        "content": {"status": "ok"},
                    }
                )
            )
            client._dispatch(
                json.dumps(
                    {
                        "channel": "iopub",
                        "header": {"msg_type": "status"},
                        "parent_header": {"msg_id": parent_id},
                        "content": {"execution_state": "idle"},
                    }
                )
            )

    client._ws = ActiveWebSocket()
    result = client.execute("work()", timeout=1.0)
    assert [item["text"] for item in result["outputs"]] == ["one", "two"]
