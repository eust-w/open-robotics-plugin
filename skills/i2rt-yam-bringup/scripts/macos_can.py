# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portable macOS CAN bus bridge for i2rt (gs_usb, PCAN, SLCAN) without sudo."""

from __future__ import annotations

import atexit
from collections import deque
import ctypes.util
import importlib
import os
import platform
import re
import threading
import time
from typing import Any

import can

_INSTALLED = False
_HEALTH_CHECKED_KEYS: set[tuple[int, int]] = set()


def _libusb_package_candidates() -> tuple[str, ...]:
  """Returns the bundled dylib path from `libusb-package` if installed."""
  try:
    pkg = importlib.import_module("libusb_package")
  except ImportError:
    return ()
  path = str(pkg.get_library_path())
  return (path,) if path else ()


_MACOS_LIBUSB_CANDIDATES = (
    "/opt/homebrew/lib/libusb-1.0.dylib",  # Apple Silicon Homebrew
    "/usr/local/lib/libusb-1.0.dylib",  # Intel Homebrew
    "/opt/local/lib/libusb-1.0.dylib",  # MacPorts
    *_libusb_package_candidates(),
)


def _ensure_macos_libusb() -> Any:
    """Ensures pyusb finds libusb-1.0 on Apple Silicon & Intel Macs without DYLD_LIBRARY_PATH."""
    from usb.backend import libusb1

    backend = libusb1.get_backend()
    if backend is not None:
        return backend

    def _find_libusb(name: str) -> str | None:
        found = ctypes.util.find_library(name)
        if found:
            return found
        for path in _MACOS_LIBUSB_CANDIDATES:
            if os.path.exists(path):
                return path
        return None

    backend = libusb1.get_backend(find_library=_find_libusb)
    if backend is None:
        raise RuntimeError(
            "libusb-1.0 not found on macOS. Install via Homebrew: `brew install libusb`."
        )
    # Patch libusb1.get_backend so libraries calling it with no args (e.g. gs_usb) succeed.
    orig_get_backend = libusb1.get_backend
    libusb1.get_backend = lambda find_library=None: orig_get_backend(
        find_library=find_library or _find_libusb
    )
    return backend


def _leave_stm32_dfu_if_present(backend: Any) -> bool:
    """If an STM32 ROM DFU device (0483:df11) is waiting on USB, boots it to 0x08000000."""
    import usb.core

    dfu_devs = list(
        usb.core.find(find_all=True, idVendor=0x0483, idProduct=0xDF11, backend=backend)
    )
    if not dfu_devs:
        return False
    for d_dfu in dfu_devs:
        try:
            st = d_dfu.ctrl_transfer(0xA1, 3, 0, 0, 6)
            if st[4] == 10:  # dfuERROR -> DFU_CLRSTATUS
                d_dfu.ctrl_transfer(0x21, 4, 0, 0, None)
                d_dfu.ctrl_transfer(0xA1, 3, 0, 0, 6)
            # Set Address Pointer to 0x08000000 and advance state dfuDNBUSY -> dfuDNLOAD-IDLE
            d_dfu.ctrl_transfer(0x21, 1, 0, 0, bytes([0x21, 0x00, 0x00, 0x00, 0x08]))
            d_dfu.ctrl_transfer(0xA1, 3, 0, 0, 6)
            d_dfu.ctrl_transfer(0xA1, 3, 0, 0, 6)
            # 0-byte DFU_DNLOAD + GETSTATUS triggers leave/manifest reset
            d_dfu.ctrl_transfer(0x21, 1, 0, 0, b"")
            try:
                d_dfu.ctrl_transfer(0xA1, 3, 0, 0, 6)
            except Exception:
                pass
        except Exception:
            pass
    time.sleep(2.0)
    return True


def _find_dfu_interface_number(usb_dev: Any) -> int | None:
    """Returns the bInterfaceNumber of a USB DFU runtime interface if exposed."""
    try:
        cfg = usb_dev.get_active_configuration()
        for intf in cfg:
            if intf.bInterfaceClass == 0xFE and intf.bInterfaceSubClass == 0x01:
                return int(intf.bInterfaceNumber)
    except Exception:
        pass
    return None


def _scan_sorted_gs_usb() -> list[Any]:
    """Scans gs_usb adapters sorted deterministically by (bus, address) for multi-arm stations."""
    from gs_usb.gs_usb import GsUsb

    devs = list(GsUsb.scan())
    devs.sort(key=lambda d: (int(d.gs_usb.bus or 0), int(d.gs_usb.address or 0)))
    return devs


def _parse_adapter_index(channel: Any) -> int:
    """Maps 'can0' -> 0, 'can1' -> 1, 0 -> 0, etc."""
    if isinstance(channel, int):
        return max(0, channel)
    m = re.search(r"(\d+)$", str(channel))
    return int(m.group(1)) if m else 0


def _ensure_gs_usb_healthy(idx: int, bitrate: int, backend: Any) -> None:
    """Motor-independent loopback check on EP 0x81; DFU-reboots only if wedged."""
    import usb.util
    from gs_usb.gs_usb import (
        DeviceMode,
        GS_CAN_MODE_HW_TIMESTAMP,
        GS_CAN_MODE_LOOP_BACK,
        GS_CAN_MODE_NORMAL,
        GS_CAN_MODE_START,
        _GS_USB_BREQ_MODE,
    )
    from gs_usb.gs_usb_frame import GsUsbFrame

    _leave_stm32_dfu_if_present(backend)
    devs = _scan_sorted_gs_usb()
    if idx >= len(devs):
        raise can.CanInitializationError(
            f"Requested CAN adapter index {idx}, but found {len(devs)} gs_usb device(s)."
        )
    g = devs[idx]
    key = (int(g.gs_usb.bus or 0), int(g.gs_usb.address or 0))
    if key in _HEALTH_CHECKED_KEYS:
        return

    # Silent internal loopback probe (never transmits onto physical CAN wires).
    if not (g.device_capability.feature & GS_CAN_MODE_LOOP_BACK):
        _HEALTH_CHECKED_KEYS.add(key)
        return

    usb.util.claim_interface(g.gs_usb, 0)
    ok_ep81 = False
    try:
        bt = can.BitTiming.from_sample_point(
            f_clock=g.device_capability.fclk_can, bitrate=bitrate, sample_point=87.5
        )
        g.set_timing(
            prop_seg=1,
            phase_seg1=bt.tseg1 - 1,
            phase_seg2=bt.tseg2,
            sjw=bt.sjw,
            brp=bt.brp,
        )
        flags = (
            GS_CAN_MODE_LOOP_BACK | GS_CAN_MODE_HW_TIMESTAMP
        ) & g.device_capability.feature
        g.device_flags = flags
        g.gs_usb.ctrl_transfer(
            0x41, _GS_USB_BREQ_MODE, 0, 0, DeviceMode(GS_CAN_MODE_START, flags).pack()
        )
        g.send(GsUsbFrame(can_id=0x7FF, data=[0] * 8))
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.03:
            try:
                if g.read(frame=GsUsbFrame(), timeout_ms=2):
                    ok_ep81 = True
                    break
            except Exception:
                break
        g.stop()
        while True:
            try:
                if not g.read(frame=GsUsbFrame(), timeout_ms=2):
                    break
            except Exception:
                break
    finally:
        try:
            usb.util.release_interface(g.gs_usb, 0)
        except Exception:
            pass

    if ok_ep81:
        _HEALTH_CHECKED_KEYS.add(key)
        return

    # EP 0x81 is wedged from a prior unclosed reset; reboot via runtime DFU if supported.
    dfu_intf = _find_dfu_interface_number(g.gs_usb)
    if dfu_intf is not None:
        try:
            g.gs_usb.ctrl_transfer(0x21, 0, 1000, dfu_intf, None)
        except Exception:
            pass
        time.sleep(1.5)
        _leave_stm32_dfu_if_present(backend)
        refreshed = _scan_sorted_gs_usb()
        if idx < len(refreshed):
            new_key = (
                int(refreshed[idx].gs_usb.bus or 0),
                int(refreshed[idx].gs_usb.address or 0),
            )
            _HEALTH_CHECKED_KEYS.add(new_key)


class MacGsUsbBus(can.BusABC):
    """Thread-safe macOS gs_usb Bus avoiding detach_kernel_driver/reset and TX echoes."""

    def __init__(self, channel: Any = "can0", bitrate: int = 1000000, **kwargs: Any) -> None:
        import usb.util
        from gs_usb.gs_usb import (
            DeviceMode,
            GS_CAN_MODE_HW_TIMESTAMP,
            GS_CAN_MODE_NORMAL,
            GS_CAN_MODE_START,
            _GS_USB_BREQ_MODE,
        )
        from gs_usb.gs_usb_frame import GS_USB_NONE_ECHO_ID, GsUsbFrame

        self._GsUsbFrame = GsUsbFrame
        self._NONE_ECHO_ID = GS_USB_NONE_ECHO_ID
        self._usb_util = usb.util

        backend = _ensure_macos_libusb()
        idx = _parse_adapter_index(channel)
        _ensure_gs_usb_healthy(idx, bitrate, backend)

        devs = _scan_sorted_gs_usb()
        if idx >= len(devs):
            raise can.CanInitializationError(
                f"Cannot open {channel!r}: found {len(devs)} gs_usb device(s)."
            )
        self.gs_usb = devs[idx]
        self.dev = self.gs_usb.gs_usb
        self.channel_info = str(channel)
        self._lock = threading.Lock()
        self._rx_queue: deque[can.Message] = deque()
        self._closed = False

        usb.util.claim_interface(self.dev, 0)
        bt = can.BitTiming.from_sample_point(
            f_clock=self.gs_usb.device_capability.fclk_can,
            bitrate=bitrate,
            sample_point=87.5,
        )
        self.gs_usb.set_timing(
            prop_seg=1,
            phase_seg1=bt.tseg1 - 1,
            phase_seg2=bt.tseg2,
            sjw=bt.sjw,
            brp=bt.brp,
        )
        flags = (
            GS_CAN_MODE_NORMAL | GS_CAN_MODE_HW_TIMESTAMP
        ) & self.gs_usb.device_capability.feature
        self.gs_usb.device_flags = flags
        self.dev.ctrl_transfer(
            0x41, _GS_USB_BREQ_MODE, 0, 0, DeviceMode(GS_CAN_MODE_START, flags).pack()
        )
        while self.gs_usb.read(frame=GsUsbFrame(), timeout_ms=2):
            pass
        atexit.register(self.shutdown)
        super().__init__(channel=channel, **kwargs)

    def send(self, msg: can.Message, timeout: float | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            frame = self._GsUsbFrame(can_id=msg.arbitration_id, data=list(msg.data))
            self.gs_usb.send(frame)

    def _recv_internal(
        self, timeout: float | None
    ) -> tuple[can.Message | None, bool]:
        wait_s = 0.010 if timeout is None else max(0.008, float(timeout))
        deadline = time.monotonic() + wait_s
        while True:
            with self._lock:
                if self._closed:
                    return None, False
                if self._rx_queue:
                    return self._rx_queue.popleft(), False
                rf = self._GsUsbFrame()
                try:
                    ok = self.gs_usb.read(frame=rf, timeout_ms=2)
                except Exception:
                    ok = False
            if ok:
                if rf.echo_id != self._NONE_ECHO_ID:
                    continue
                rx_msg = can.Message(
                    timestamp=rf.timestamp_us / 1e6,
                    arbitration_id=rf.can_id & 0x7FF,
                    is_extended_id=False,
                    dlc=rf.can_dlc,
                    data=bytes(rf.data[: rf.can_dlc]),
                )
                return rx_msg, False
            if time.monotonic() >= deadline:
                return None, False

    def shutdown(self) -> None:
        if self._closed:
            return
        # Allow DMChainCanInterface control thread to observe running=False before stop.
        time.sleep(0.04)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.gs_usb.stop()
                while self.gs_usb.read(frame=self._GsUsbFrame(), timeout_ms=2):
                    pass
            except Exception:
                pass
            try:
                self._usb_util.release_interface(self.dev, 0)
            except Exception:
                pass
        super().shutdown()


def _patch_i2rt_clean_error_drain() -> None:
    """Ensures DMSingleMotorCanInterface.clean_error drains all 3 0xFB replies."""
    try:
        from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
    except ImportError:
        return
    if getattr(DMSingleMotorCanInterface, "_macos_clean_error_patched", False):
        return
    orig_clean_error = DMSingleMotorCanInterface.clean_error

    def _clean_error_and_drain(self: Any, motor_id: int) -> None:
        orig_clean_error(self, motor_id)
        self._drain_bus(timeout_s=0.02, idle_count=4)

    DMSingleMotorCanInterface.clean_error = _clean_error_and_drain
    DMSingleMotorCanInterface._macos_clean_error_patched = True


def install_mac_can(
    interface: str | None = None,
    backend_channel: str | int | None = None,
    bitrate: int = 1000000,
) -> None:
    """Bridges i2rt's socketcan ('can0', 'can1', ...) onto macOS USB CAN adapters.

    Safe no-op on Linux. On macOS, auto-detects `gs_usb` adapters (candleLight /
    CANable / I2RT USB-CAN) or honors `interface` / `I2RT_MAC_CAN_INTERFACE`
    (e.g. `'gs_usb'`, `'pcan'`, `'slcan'`, `'kvaser'`) and `backend_channel` /
    `I2RT_MAC_CAN_CHANNEL` (e.g. `'PCAN_USBBUS1'` or `'/dev/cu.usbmodem...'`).
    """
    global _INSTALLED
    if platform.system().lower() != "darwin" or _INSTALLED:
        return

    _patch_i2rt_clean_error_drain()
    chosen_if = (interface or os.environ.get("I2RT_MAC_CAN_INTERFACE") or "").strip().lower()
    chosen_ch = backend_channel or os.environ.get("I2RT_MAC_CAN_CHANNEL")

    orig_bus = can.interface.Bus

    def _mac_bus_factory(channel: Any = None, *args: Any, **kwargs: Any) -> can.BusABC:
        req_if = (kwargs.get("interface") or kwargs.get("bustype") or "").lower()
        is_socketcan = req_if == "socketcan" or (
            isinstance(channel, str) and channel.startswith("can")
        )
        if not is_socketcan:
            return orig_bus(channel, *args, **kwargs)

        kwargs.pop("bustype", None)
        kwargs.pop("interface", None)
        bus_bitrate = int(kwargs.pop("bitrate", bitrate))

        target_if = chosen_if
        if not target_if:
            try:
                backend = _ensure_macos_libusb()
                _leave_stm32_dfu_if_present(backend)
                if _scan_sorted_gs_usb():
                    target_if = "gs_usb"
            except Exception:
                pass
        if not target_if:
            target_if = "pcan"

        if target_if == "gs_usb":
            return MacGsUsbBus(
                channel=chosen_ch if chosen_ch is not None else (channel or "can0"),
                bitrate=bus_bitrate,
                **kwargs,
            )

        idx = _parse_adapter_index(channel)
        real_ch = chosen_ch
        if real_ch is None:
            real_ch = f"PCAN_USBBUS{idx + 1}" if target_if == "pcan" else channel
        return orig_bus(real_ch, interface=target_if, bitrate=bus_bitrate, **kwargs)

    can.interface.Bus = _mac_bus_factory
    _INSTALLED = True
