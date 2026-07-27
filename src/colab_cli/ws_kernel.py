# ws_kernel.py — thay thế jupyter_kernel_client bằng WebSocket thuần.
#
# Lý do tồn tại file này: jupyter_kernel_client kéo theo nbformat -> jsonschema
# -> referencing -> rpds-py (Rust/maturin), không có wheel prebuilt cho
# Android/Termux arm64. File này implement lại đúng phần API surface mà
# colab-cli thực sự dùng (execute_code, restart, stop, colab_request_hook)
# bằng `websocket-client` (thuần Python, đã là dependency sẵn có) + giao
# thức Jupyter kernel wire protocol v5 (JSON, không cần HMAC vì Colab dùng
# token trong header thay vì ký message — xác nhận qua behavior thực tế của
# server; nếu Colab đổi sang yêu cầu chữ ký, xem ghi chú ở _sign()).
#
# Tham khảo giao thức: https://jupyter-client.readthedocs.io/en/stable/messaging.html

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode, urljoin

import websocket  # websocket-client package

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "5.3"
DEFAULT_USERNAME = "colab-cli"


def _new_msg_id() -> str:
    return str(uuid.uuid4())


def _make_header(msg_type: str, session: str) -> Dict[str, Any]:
    return {
        "msg_id": _new_msg_id(),
        "session": session,
        "username": DEFAULT_USERNAME,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "msg_type": msg_type,
        "version": PROTOCOL_VERSION,
    }


class KernelStartError(Exception):
    pass


class WSKernelClient:
    """Minimal Jupyter kernel client over a raw WebSocket.

    Public surface mirrors what colab-cli's runtime.py actually calls on
    jupyter_kernel_client.KernelClient, so runtime.py needs only import
    changes, not logic changes:
      - .id (kernel_id)
      - .start()
      - .execute(code, allow_stdin=..., timeout=...) -> dict with 'outputs'
      - .execute_interactive(code, output_hook=..., allow_stdin=..., stdin_hook=..., timeout=...)
      - .restart(timeout=...)
      - ._manager.client.stop_channels()
      - ._manager.client.kernel_socket.close()
      - ._manager.shutdown_kernel(now=True)
      - ._manager.client.session.session  (session id string)
      - ._manager.client.kernel_socket.on_message  (patchable, for colab_request hook)
      - ._manager.client.session.msg(msg_type, content)  (used by automation.py to build input_reply)
      - ._manager.client.stdin_channel.send(msg)
    """

    def __init__(
        self,
        server_url: str,
        token: str,
        kernel_id: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.id = kernel_id
        self.session_id = None
        if client_kwargs and client_kwargs.get("session"):
            self.session_id = client_kwargs["session"]
        self.extra_headers = headers or {}
        self._own_kernel = True

        self._ws: Optional[websocket.WebSocket] = None
        self._ws_lock = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # Queues keyed by parent msg_id, fed by the receive loop.
        self._iopub_queues: Dict[str, "queue.Queue[dict]"] = {}
        self._shell_replies: Dict[str, dict] = {}
        self._shell_events: Dict[str, threading.Event] = {}
        self._stdin_waiters: Dict[str, "queue.Queue[dict]"] = {}

        # Exposed so runtime.py's _apply_ws_hook can patch it, and so
        # automation.py's drivefs_hook can be invoked with a wsclient-like
        # object.
        self.on_message: Optional[Callable[[Any, str], None]] = None

        # Shim objects so runtime.py's `_manager.client...` chain keeps working
        self._manager = _ManagerShim(self)
        self.session = _SessionShim(self)

    # -- lifecycle -----------------------------------------------------

    def start(self, timeout: float = 60.0):
        if not self.id:
            self.id = self._create_kernel(timeout=timeout)

        if not self.session_id:
            self.session_id = str(uuid.uuid4())

        ws_url = self._build_ws_url()
        headers = [f"{k}: {v}" for k, v in self.extra_headers.items()]

        self._ws = websocket.create_connection(
            ws_url, header=headers, timeout=timeout
        )
        self._stop_flag.clear()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _create_kernel(self, timeout: float) -> str:
        """POST /api/kernels to start a new kernel, return its id."""
        import requests

        url = urljoin(self.server_url + "/", "api/kernels")
        headers = dict(self.extra_headers)
        headers["Authorization"] = f"token {self.token}"
        resp = requests.post(url, headers=headers, json={}, timeout=timeout)
        if not resp.ok:
            raise KernelStartError(
                f"Failed to start kernel: {resp.status_code} {resp.text}"
            )
        return resp.json()["id"]

    def _build_ws_url(self) -> str:
        base = self.server_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        params = {"session_id": self.session_id}
        return urljoin(
            base + "/", f"api/kernels/{self.id}/channels?{urlencode(params)}"
        )

    def stop_channels(self):
        self._stop_flag.set()
        with self._ws_lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass

    def shutdown_kernel(self, now: bool = True):
        import requests

        url = urljoin(self.server_url + "/", f"api/kernels/{self.id}")
        headers = dict(self.extra_headers)
        headers["Authorization"] = f"token {self.token}"
        try:
            requests.delete(url, headers=headers, timeout=10)
        except Exception:
            logger.debug("Failed to delete kernel on shutdown", exc_info=True)

    # -- message plumbing -----------------------------------------------

    def _send(self, channel: str, msg_type: str, content: Dict[str, Any]) -> str:
        header = _make_header(msg_type, self.session_id)
        msg = {
            "header": header,
            "parent_header": {},
            "metadata": {},
            "content": content,
            "channel": channel,
        }
        with self._ws_lock:
            self._ws.send(json.dumps(msg))
        return header["msg_id"]

    def _recv_loop(self):
        while not self._stop_flag.is_set():
            try:
                raw = self._ws.recv()
            except Exception:
                if self._stop_flag.is_set():
                    return
                logger.debug("WS recv error", exc_info=True)
                return
            if not raw:
                continue

            # Give the pluggable hook (colab_request interception) first look,
            # exactly like the original on_message patch point.
            if self.on_message:
                try:
                    self.on_message(self, raw)
                    continue
                except _PassThrough:
                    pass  # hook declined to handle; fall through to default

            self._dispatch(raw)

    def _dispatch(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        channel = msg.get("channel")
        msg_type = msg.get("header", {}).get("msg_type")
        parent_id = msg.get("parent_header", {}).get("msg_id")

        if channel == "iopub" and parent_id:
            q = self._iopub_queues.get(parent_id)
            if q is not None:
                q.put(msg)
            if msg_type == "status" and msg.get("content", {}).get(
                "execution_state"
            ) == "idle":
                if q is not None:
                    q.put({"__idle__": True})

        elif channel == "shell" and parent_id:
            self._shell_replies[parent_id] = msg
            ev = self._shell_events.get(parent_id)
            if ev:
                ev.set()

        elif channel == "stdin" and parent_id:
            q = self._stdin_waiters.get(parent_id)
            if q is not None:
                q.put(msg)

    # -- execute ---------------------------------------------------------

    def execute(
        self,
        code: str,
        allow_stdin: bool = False,
        stdin_hook: Optional[Callable[[str], str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        outputs: List[Dict[str, Any]] = []
        reply = self._execute_common(
            code,
            allow_stdin=allow_stdin,
            stdin_hook=stdin_hook,
            output_hook=lambda o: outputs.append(o),
            timeout=timeout,
        )
        reply = dict(reply or {})
        reply["outputs"] = outputs
        return reply

    def execute_interactive(
        self,
        code: str,
        output_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        allow_stdin: bool = False,
        stdin_hook: Optional[Callable[[str], str]] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        reply = self._execute_common(
            code,
            allow_stdin=allow_stdin,
            stdin_hook=stdin_hook,
            output_hook=output_hook,
            timeout=timeout,
        )
        return reply

    def _execute_common(
        self,
        code: str,
        allow_stdin: bool,
        stdin_hook: Optional[Callable[[str], str]],
        output_hook: Optional[Callable[[Dict[str, Any]], None]],
        timeout: float,
    ) -> Dict[str, Any]:
        content = {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": allow_stdin,
            "stop_on_error": True,
        }
        msg_id = self._send("shell", "execute_request", content)

        iopub_q: "queue.Queue[dict]" = queue.Queue()
        self._iopub_queues[msg_id] = iopub_q
        shell_ev = threading.Event()
        self._shell_events[msg_id] = shell_ev
        stdin_q: "queue.Queue[dict]" = queue.Queue()
        self._stdin_waiters[msg_id] = stdin_q

        deadline = time.time() + timeout if timeout else None
        got_idle = False

        try:
            while not got_idle:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Execution timed out after {timeout}s"
                        )

                # Check stdin requests (non-blocking-ish)
                try:
                    stdin_msg = stdin_q.get_nowait()
                    prompt = stdin_msg.get("content", {}).get("prompt", "")
                    password = stdin_msg.get("content", {}).get("password", False)
                    if allow_stdin:
                        value = (
                            stdin_hook(prompt)
                            if stdin_hook
                            else input(prompt)
                        )
                    else:
                        value = ""
                    reply_id = self._send(
                        "stdin", "input_reply", {"value": value, "password": password}
                    )
                    # extend deadline a bit — user interaction shouldn't be
                    # counted against the execution timeout
                    if deadline is not None:
                        deadline = time.time() + timeout
                except queue.Empty:
                    pass

                try:
                    item = iopub_q.get(timeout=min(remaining or 1.0, 1.0))
                except queue.Empty:
                    continue

                if item.get("__idle__"):
                    got_idle = True
                    break

                out = _iopub_to_output(item)
                if out and output_hook:
                    output_hook(out)
        finally:
            self._iopub_queues.pop(msg_id, None)
            self._shell_events.pop(msg_id, None)
            self._stdin_waiters.pop(msg_id, None)

        shell_ev.wait(timeout=5)
        reply = self._shell_replies.pop(msg_id, {})
        content = reply.get("content", {"status": "ok"})
        return content

    def restart(self, timeout: Optional[float] = None):
        import requests

        url = urljoin(self.server_url + "/", f"api/kernels/{self.id}/restart")
        headers = dict(self.extra_headers)
        headers["Authorization"] = f"token {self.token}"
        resp = requests.post(url, headers=headers, timeout=timeout or 30)
        if not resp.ok:
            raise KernelStartError(
                f"Failed to restart kernel: {resp.status_code} {resp.text}"
            )


class _PassThrough(Exception):
    """Raise from an on_message hook to indicate 'not handled, continue'."""


class _ManagerShim:
    """Mimics KernelClient._manager.client... chain used by runtime.py."""

    def __init__(self, ws_client: WSKernelClient):
        self.client = _ClientShim(ws_client)

    def shutdown_kernel(self, now: bool = True):
        self.client._ws_client.shutdown_kernel(now=now)


class _ClientShim:
    def __init__(self, ws_client: WSKernelClient):
        self._ws_client = ws_client
        self.session = _SessionShim(ws_client)
        self.stdin_channel = _StdinChannelShim(ws_client)

    def stop_channels(self):
        self._ws_client.stop_channels()

    @property
    def kernel_socket(self):
        return _SocketShim(self._ws_client)


class _SocketShim:
    """Exposes .on_message as a patchable attribute, like wsclient.kernel_socket."""

    def __init__(self, ws_client: WSKernelClient):
        self._ws_client = ws_client

    @property
    def on_message(self):
        return self._ws_client.on_message

    @on_message.setter
    def on_message(self, fn):
        self._ws_client.on_message = fn

    def close(self):
        self._ws_client.stop_channels()


class _SessionShim:
    def __init__(self, ws_client: WSKernelClient):
        self._ws_client = ws_client
        self.session = ws_client.session_id

    def msg(self, msg_type: str, content: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "header": _make_header(msg_type, self._ws_client.session_id),
            "parent_header": {},
            "metadata": {},
            "content": content,
        }

    def deserialize(self, msg_list):
        # Only used for the V1 binary subprotocol, which colab-cli does not
        # select (it always uses JupyterSubprotocol.DEFAULT). Kept as a
        # clear error rather than silent wrong behavior if that ever changes.
        raise NotImplementedError(
            "V1 binary subprotocol not implemented in ws_kernel.py lite client"
        )


class _StdinChannelShim:
    def __init__(self, ws_client: WSKernelClient):
        self._ws_client = ws_client

    def send(self, msg: Dict[str, Any]):
        with self._ws_client._ws_lock:
            msg = dict(msg)
            msg["channel"] = "stdin"
            self._ws_client._ws.send(json.dumps(msg))


def _iopub_to_output(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw iopub message into the output dict shape colab-cli expects
    (output_type/text/data/ename/evalue/traceback), matching what
    jupyter_kernel_client's default output_hook produced.
    """
    msg_type = msg.get("header", {}).get("msg_type")
    content = msg.get("content", {})

    if msg_type == "stream":
        return {
            "output_type": "stream",
            "name": content.get("name", "stdout"),
            "text": content.get("text", ""),
        }
    if msg_type in ("execute_result", "display_data"):
        return {
            "output_type": msg_type,
            "data": content.get("data", {}),
            "metadata": content.get("metadata", {}),
        }
    if msg_type == "error":
        return {
            "output_type": "error",
            "ename": content.get("ename", "Error"),
            "evalue": content.get("evalue", ""),
            "traceback": content.get("traceback", []),
        }
    # status, execute_input, clear_output, etc. — not surfaced as an output
    return None
