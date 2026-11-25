# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Loopback HTTP proxy between a `DeviceLayer` and an out-of-process controller.

The controller only sees the task's real-robot key allowlist, so sim-only keys
stay invisible and sim/real key parity is enforced. Scalar/1-D readings return
eagerly on `POST /begin` and `POST /step`; 2-D/3-D camera arrays are fetched
lazily on demand via `GET /read?key=<name>` so 50 Hz proprio steps stay sub-KB.
All `DeviceLayer` calls run on the caller's thread (GL contexts are thread-bound).
"""

import base64
from collections.abc import Callable, Iterator, Mapping, Sequence
import http.server
import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from dm_env import specs
import numpy as np


class EpisodeOverError(RuntimeError):
  """Raised when the gauntlet ended the episode or hit `max_ticks`."""


class _MeasView(Mapping[str, np.ndarray]):
  """Read-only measurement view with an on-miss reader/tracker hook."""

  def __init__(self, cache: dict[str, np.ndarray], keys: tuple[str, ...], read: Callable[[str], np.ndarray]):
    self._cache, self._keys, self._set, self._read = cache, keys, frozenset(keys), read

  def __getitem__(self, key: str) -> np.ndarray:
    if key not in self._set:
      raise KeyError(key)
    return self._cache[key] if key in self._cache else self._read(key)

  def __contains__(self, key: object) -> bool:
    return key in self._set

  def __iter__(self) -> Iterator[str]:
    return iter(self._keys)

  def __len__(self) -> int:
    return len(self._keys)


class AllowlistLayer:
  """Key allowlist, command validator, tick cap, and camera-read tracker."""

  def __init__(
      self, device_layer: Any, allow: tuple[frozenset[str], frozenset[str]],
      *, max_ticks: int | None = None, tick_fn: Callable[[], int] | None = None,
  ):
    self._device_layer, self._max_ticks, self._tick_fn = device_layer, max_ticks, tick_fn
    meas_keys, cmd_keys = allow
    mspec, cspec = device_layer.measurements_spec(), device_layer.commands_spec()
    full = meas_keys | {f"{k}_timestamp" for k in meas_keys if f"{k}_timestamp" in mspec}
    missing = (full - mspec.keys()) | (cmd_keys - cspec.keys())
    if missing:
      raise ValueError(f"Deployment keys missing from the device layer: {missing}")
    self._keys = tuple(sorted(full))
    self._mspec = {k: mspec[k] for k in self._keys}
    self._cspec = {k: cspec[k] for k in sorted(cmd_keys)}
    self._cams = frozenset(k for k, s in self._mspec.items() if len(s.shape) >= 2)
    self.ticks, self.deadline, self.last_cam_read_tick = 0, False, -1
    self.begun = self.ended = self._layer_ended = False

  def measurements_spec(self) -> Mapping[str, specs.Array]:
    return self._mspec

  def commands_spec(self) -> Mapping[str, specs.Array]:
    return self._cspec

  def _view(self, raw: Mapping[str, Any]) -> _MeasView:
    tick = self._tick_fn() if self._tick_fn is not None else self.ticks
    def _read(k: str) -> np.ndarray:
      if k in self._cams:
        self.last_cam_read_tick = max(self.last_cam_read_tick, tick)
      return np.array(raw[k], copy=True)
    return _MeasView({}, self._keys, _read)

  def parse_commands(self, commands: Mapping[str, Any]) -> tuple[dict[str, np.ndarray] | None, str]:
    if set(commands) != self._cspec.keys():
      return None, f"commands must be exactly {sorted(self._cspec)}"
    cmd = {k: np.asarray(v, dtype=self._cspec[k].dtype) for k, v in commands.items()}
    for k, v in cmd.items():
      if v.shape != tuple(self._cspec[k].shape) or not np.all(np.isfinite(v)):
        return None, f"bad value for {k}"
    return cmd, ""

  def begin_stepping(self) -> Mapping[str, np.ndarray]:
    if self.begun or self.ended:
      raise EpisodeOverError("episode over")
    self.begun = True
    return self._view(self._device_layer.begin_stepping())

  def step(self, commands: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    if self.ended or not self.begun:
      raise EpisodeOverError("episode over")
    if self._max_ticks is not None and self.ticks >= self._max_ticks:
      self.deadline = True
      self.end_stepping()
      raise EpisodeOverError("deadline")
    cmd, err = self.parse_commands(commands)
    if cmd is None:
      raise ValueError(err)
    self.ticks += 1
    return self._view(self._device_layer.step(cmd))

  def end_stepping(self) -> None:
    self.ended = True
    if self.begun and not self._layer_ended:
      self._layer_ended = True
      self._device_layer.end_stepping()


def encode(value: Any) -> dict[str, Any]:
  arr = np.asarray(value)
  return {
      "dtype": arr.dtype.str,
      "shape": list(arr.shape),
      "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
  }


def decode(doc: Mapping[str, Any]) -> np.ndarray:
  raw = base64.b64decode(doc["b64"])
  return np.frombuffer(raw, np.dtype(doc["dtype"])).reshape(doc["shape"]).copy()


def _spec_to_json(s: specs.Array) -> dict[str, Any]:
  doc: dict[str, Any] = {"shape": list(s.shape), "dtype": np.dtype(s.dtype).str}
  if isinstance(s, specs.BoundedArray):
    doc["min"], doc["max"] = (
        np.asarray(s.minimum).tolist(), np.asarray(s.maximum).tolist()
    )
  return doc


def _spec_from_json(doc: Mapping[str, Any]) -> specs.Array:
  shape, dtype = tuple(doc["shape"]), np.dtype(doc["dtype"])
  if "min" in doc:
    return specs.BoundedArray(shape, dtype, doc["min"], doc["max"])
  return specs.Array(shape, dtype)


def launch_controller(
    cmd: Sequence[str],
    url: str,
    env: Mapping[str, str] | None = None,
    *,
    prompt: str = "",
) -> tuple[subprocess.Popen[bytes], Any, Any]:
  """Starts the controller in its own process group with a scrubbed env."""
  base = {k: os.environ[k] for k in ("PATH", "HOME", "MUJOCO_GL") if k in os.environ}
  clean = {
      k: v for k, v in (env or {}).items()
      if not k.startswith(("ROBOSIM_", "GAUNTLET_"))
  }
  extra = {"GAUNTLET_PROMPT": prompt} if prompt else {}
  out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
  proc = subprocess.Popen(
      list(cmd),
      env={**base, **clean, "GAUNTLET_URL": url, **extra},
      stdout=out,
      stderr=err,
      stdin=subprocess.DEVNULL,
      start_new_session=True,
  )
  return proc, out, err


def _outcome(proc: Any, out: Any, err: Any, timed_out: bool) -> dict[str, Any]:
  """Parses the last stdout line `{"status": "done"|"abort", "fault": ...}`."""
  with out, err:
    out.seek(0)
    err.seek(0)
    lines = [l for l in out.read().decode(errors="replace").splitlines() if l.strip()]
    doc: dict[str, Any] = {"status": "timeout" if timed_out else "crash", "fault": ""}
    if not timed_out and lines and lines[-1].lstrip().startswith("{"):
      try:
        last = json.loads(lines[-1])
      except json.JSONDecodeError:
        last = None
      if isinstance(last, Mapping) and last.get("status") in ("done", "abort"):
        doc = {"status": last["status"], "fault": str(last.get("fault", ""))}
    doc["returncode"] = proc.returncode
    doc["stderr_tail"] = err.read().decode(errors="replace")[-2000:]
  return doc


def serve(
    device_layer: Any,
    controller_cmd: Sequence[str],
    *,
    allow: tuple[frozenset[str], frozenset[str]],
    max_ticks: int,
    wall_timeout_s: float,
    env: Mapping[str, str] | None = None,
    prompt: str = "",
    tick_fn: Callable[[], int] | None = None,
) -> dict[str, Any]:
  """Runs one controller episode against `device_layer`; returns its outcome."""
  client = AllowlistLayer(device_layer, allow, max_ticks=max_ticks, tick_fn=tick_fn)
  mspec, cspec = client.measurements_spec(), client.commands_spec()
  eager_keys = [k for k, s in mspec.items() if len(s.shape) < 2]
  spec_doc = {
      "measurements": {k: _spec_to_json(s) for k, s in mspec.items()},
      "commands": {k: _spec_to_json(s) for k, s in cspec.items()},
  }
  requests: queue.Queue[Any] = queue.Queue()

  class Handler(http.server.BaseHTTPRequestHandler):

    def _relay(self) -> None:
      size = int(self.headers.get("Content-Length", 0))
      body = json.loads(self.rfile.read(size) or b"{}")
      reply: queue.Queue[Any] = queue.Queue()
      requests.put((self.command, self.path, body, reply))
      code, payload = reply.get(timeout=wall_timeout_s)
      data = json.dumps(payload).encode()
      self.send_response(code)
      self.send_header("Content-Length", str(len(data)))
      self.end_headers()
      self.wfile.write(data)

    do_GET = do_POST = _relay

    def log_message(self, *args: Any) -> None:
      del args

  server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  threading.Thread(target=server.serve_forever, daemon=True).start()
  url = f"http://127.0.0.1:{server.server_address[1]}"
  proc, out, err = launch_controller(controller_cmd, url, env, prompt=prompt)
  cur: list[Mapping[str, np.ndarray] | None] = [None]

  def view(meas: Mapping[str, np.ndarray]) -> dict[str, Any]:
    cur[0] = meas
    return {k: encode(meas[k]) for k in eager_keys}

  def handle(method: str, path: str, body: Any) -> tuple[int, Any]:
    parsed = urllib.parse.urlsplit(path)
    route = parsed.path
    if method == "GET" and route == "/specs":
      return 200, spec_doc
    if client.ended:
      return 409, {"error": "episode over"}
    if method == "GET" and route == "/read":
      if not client.begun or cur[0] is None:
        return 400, {"error": "not begun"}
      key = urllib.parse.parse_qs(parsed.query).get("key", [""])[0]
      if key not in mspec:
        return 403, {"error": f"forbidden key {key!r}"}
      return 200, encode(cur[0][key])
    if route == "/begin" and not client.begun:
      return 200, view(client.begin_stepping())
    if route == "/end":
      client.end_stepping()
      return 200, {}
    if route != "/step" or not client.begun:
      return 400, {"error": f"bad request {method} {path}"}
    if client.ticks >= max_ticks:
      client.deadline = True
      client.end_stepping()
      return 409, {"error": "deadline"}
    if set(body) != cspec.keys():
      return 400, {"error": f"commands must be exactly {sorted(cspec)}"}
    cmd, err_msg = client.parse_commands({k: decode(doc) for k, doc in body.items()})
    if cmd is None:
      return 400, {"error": err_msg}
    return 200, view(client.step(cmd))

  t_end, timed_out = time.monotonic() + wall_timeout_s, False
  try:
    while True:
      try:
        method, path, body, reply = requests.get(timeout=0.05)
      except queue.Empty:
        if proc.poll() is not None:
          break
        if time.monotonic() > t_end:
          timed_out = True
          break
        continue
      reply.put(handle(method, path, body))
  finally:
    if client.begun:
      client.end_stepping()
    if proc.poll() is None:
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
    proc.wait()
    server.shutdown()
    server.server_close()
  doc = _outcome(proc, out, err, timed_out)
  doc.update(ticks=client.ticks, deadline=client.deadline, last_cam_read_tick=client.last_cam_read_tick)
  return doc


class RemoteDeviceLayer:
  """The controller-side client: a plain numpy `DeviceLayer` over the proxy."""

  def __init__(self, url: str, timeout_s: float = 120.0):
    self._url, self._timeout = url.rstrip("/"), timeout_s
    doc = self._call("GET", "/specs")
    self._mspec = {k: _spec_from_json(v) for k, v in doc["measurements"].items()}
    self._cspec = {k: _spec_from_json(v) for k, v in doc["commands"].items()}
    self._keys = tuple(self._mspec)

  def _call(self, method: str, path: str, body: Any = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(self._url + path, data=data, method=method)
    try:
      with urllib.request.urlopen(req, timeout=self._timeout) as resp:
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
      if e.code == 409:
        raise EpisodeOverError(e.read().decode()) from e
      raise

  def _read_key(self, key: str) -> np.ndarray:
    return decode(self._call("GET", f"/read?{urllib.parse.urlencode({'key': key})}"))

  def _wrap(self, raw: Mapping[str, Any]) -> _MeasView:
    eager = {k: decode(v) for k, v in raw.items()}
    return _MeasView(eager, self._keys, lambda k: eager.setdefault(k, self._read_key(k)))

  def measurements_spec(self) -> Mapping[str, specs.Array]:
    return self._mspec

  def commands_spec(self) -> Mapping[str, specs.Array]:
    return self._cspec

  def begin_stepping(self) -> Mapping[str, np.ndarray]:
    return self._wrap(self._call("POST", "/begin", {}))

  def step(self, commands: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    body = {
        k: encode(np.asarray(v, self._cspec[k].dtype)) for k, v in commands.items()
    }
    return self._wrap(self._call("POST", "/step", body))

  def end_stepping(self) -> None:
    self._call("POST", "/end", {})
