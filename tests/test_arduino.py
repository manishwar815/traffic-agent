from traffic_agent.adapters.signal.arduino_serial import ArduinoSignal, encode
from traffic_agent.config import AppConfig, ArduinoCfg
from traffic_agent.core.events import EventBus, EventType
from traffic_agent.core.pipeline import Pipeline

N_GREEN = {"N": "GREEN", "E": "RED", "S": "RED", "W": "RED"}


class FakePort:
    def __init__(self):
        self.lines, self.fail, self.closed = [], False, False

    def write(self, b):
        if self.fail:
            raise OSError("device disconnected")
        self.lines.append(b)

    def close(self):
        self.closed = True


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def _signal(opener, clock):
    bus = EventBus()
    sys_msgs = []
    bus.subscribe(lambda e: sys_msgs.append(e.message) if e.type == EventType.SYSTEM else None)
    return ArduinoSignal(ArduinoCfg(enabled=True, port="fake"), bus, opener=opener, clock=clock), sys_msgs


def test_encode_order_missing_lamps_and_two_green_failsafe():
    assert encode(N_GREEN) == b"GRRR\n"
    assert encode({"E": "AMBER", "W": "RED"}) == b"-A-R\n"
    assert encode({"N": "GREEN", "E": "GREEN", "S": "RED", "W": "RED"}) == b"RRRR\n"


def test_sends_on_change_and_heartbeat_only():
    port, clock = FakePort(), Clock()
    sig, _ = _signal(lambda p, b: port, clock)
    sig.apply(N_GREEN)
    clock.t = 0.2
    sig.apply(N_GREEN)                              # same state, before heartbeat: not sent
    clock.t = 0.3
    sig.apply(dict(N_GREEN, N="AMBER"))             # change: sent at once
    clock.t = 0.9
    sig.apply(dict(N_GREEN, N="AMBER"))             # heartbeat due
    assert port.lines == [b"GRRR\n", b"ARRR\n", b"ARRR\n"]


def test_missing_board_warns_once_retries_and_recovers_after_unplug():
    clock, ports, tries = Clock(), [], []

    def opener(p, b):
        tries.append(clock.t)
        if clock.t < 10:
            raise OSError("no such device")
        ports.append(FakePort())
        return ports[-1]

    sig, msgs = _signal(opener, clock)
    for k in range(100):                            # 10 s with no board, 10 Hz
        clock.t = k * 0.1
        sig.apply(N_GREEN)
    assert len(tries) == 2 and sum("not connected" in m for m in msgs) == 1
    clock.t = 10.0
    sig.apply(N_GREEN)
    assert ports[0].lines == [b"GRRR\n"] and "Arduino connected on fake" in msgs
    ports[0].fail = True                            # unplugged mid-run
    clock.t = 11.0
    sig.apply(N_GREEN)
    assert ports[0].closed and any("lost" in m for m in msgs)
    clock.t = 16.0
    sig.apply(N_GREEN)
    assert ports[1].lines == [b"GRRR\n"]
    assert sig.last == N_GREEN                      # the dashboard state kept updating throughout


def test_real_pyserial_loopback():
    sig = ArduinoSignal(ArduinoCfg(enabled=True, port="loop://"), EventBus())
    sig.apply(N_GREEN)
    assert sig._port.read(16) == b"GRRR\n"
    sig.close()


def test_pipeline_uses_arduino_when_enabled():
    cfg = AppConfig()
    cfg.sim.realtime = False
    cfg.signal.arduino = ArduinoCfg(enabled=True, port="loop://")
    p = Pipeline(cfg, EventBus())
    for _ in range(20):
        p.tick()
    assert isinstance(p.actuator, ArduinoSignal) and p.actuator._port.read(64).endswith(b"\n")
    p.stop()
    assert p.actuator._port is None
