"""Built-in 4-arm junction traffic simulator (left-hand traffic, as in India).

Why it exists:
  * Recorded video cannot react to OUR signal decisions. In the simulator,
    vehicles obey whatever signal the controller sets, so adaptive timing and
    emergency green can be measured for real (closed loop).
  * Rare events (ambulance, crash, red-light jump) can be injected on demand,
    which makes a live demo reliable and repeatable.

Geometry (metres): junction centre at (0,0), x = east, y = north.
Each approach has one inbound lane and one outbound lane. All vehicles go
straight through (no turns) to keep it simple.
Car following uses the Intelligent Driver Model (IDM), a standard model in
traffic research: accelerate toward a desired speed, brake for the vehicle
(or red stop line) ahead.
"""
from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

APPROACH_DIR = {          # direction of travel for INBOUND vehicles of each approach
    "N": (0.0, -1.0),     # comes from the north, drives south
    "S": (0.0, 1.0),
    "E": (-1.0, 0.0),
    "W": (1.0, 0.0),
}
LANE_W = 3.5              # metres
BOX_HALF = LANE_W         # junction box half-size (one lane each way = 3.5 m from centre)
STOP_GAP = 2.5            # stop line distance before the box edge (zebra crossing in between)

KIND_SPECS = {            # length, width, desired speed factor
    "car": (4.4, 1.8, 1.0),
    "motorcycle": (2.0, 0.8, 1.0),
    "bus": (11.0, 2.5, 0.8),
    "truck": (8.0, 2.4, 0.8),
    "ambulance": (5.6, 2.1, 1.1),
}
CAR_COLORS = [(200, 200, 200), (40, 40, 160), (160, 70, 30), (30, 30, 30), (60, 140, 220),
              (90, 90, 90), (40, 120, 40), (20, 160, 200), (180, 180, 60)]  # BGR

# IDM parameters
A_MAX = 2.0      # m/s^2 max acceleration
B_COMF = 3.0     # m/s^2 comfortable braking
S0 = 2.0         # m minimum standstill gap
T_HEAD = 1.2     # s desired time headway


def left_of(d: tuple[float, float]) -> tuple[float, float]:
    return (-d[1], d[0])


def random_plate(rng: random.Random) -> str:
    """Telangana-style plate, e.g. TS09EC6993."""
    return "TS%02d%s%04d" % (rng.randint(1, 38),
                             "".join(rng.choice(string.ascii_uppercase) for _ in range(2)),
                             rng.randint(1, 9999))


@dataclass
class SimVehicle:
    vid: int
    kind: str
    approach: str
    s: float                  # distance of the FRONT bumper along the path (m)
    v: float                  # speed m/s
    v0: float                 # desired speed m/s
    length: float
    width: float
    color: tuple[int, int, int]
    plate: str
    t_spawn: float
    wrong_way: bool = False   # drives against traffic on the outbound lane
    helmet: Optional[bool] = None   # motorcycles only
    siren: bool = False       # emergency lights/siren on
    ignore_red: bool = False  # red-light jumper
    distracted: bool = False  # does not brake for vehicles ahead (causes rear-end crash)
    speeding: bool = False    # wants to drive well above the speed limit (and accelerates hard)
    stalled: bool = False     # broken down in its lane (crash demo)
    crashed: bool = False
    crash_t: Optional[float] = None
    wait_s: float = 0.0       # time spent nearly stopped before the stop line
    crossed_stop: bool = False
    done: bool = False

    # ------------------------------------------------------------ geometry
    @property
    def dir(self) -> tuple[float, float]:
        return APPROACH_DIR[self.approach]

    def lane_offset(self) -> tuple[float, float]:
        lx, ly = left_of(self.dir)
        k = -LANE_W / 2 if self.wrong_way else LANE_W / 2   # left-hand traffic
        return lx * k, ly * k

    def front_xy(self, arm_len: float) -> tuple[float, float]:
        dx, dy = self.dir
        ox, oy = self.lane_offset()
        return (-dx * arm_len + ox + dx * self.s, -dy * arm_len + oy + dy * self.s)

    def footprint(self, arm_len: float) -> tuple[float, float, float, float]:
        """Axis-aligned rectangle (xmin, ymin, xmax, ymax) in metres.
        All paths are straight N-S or E-W, so footprints are always axis-aligned."""
        fx, fy = self.front_xy(arm_len)
        dx, dy = self.dir
        rx, ry = fx - dx * self.length, fy - dy * self.length   # rear centre
        lx, ly = left_of(self.dir)
        hw = self.width / 2
        xs = [fx + lx * hw, fx - lx * hw, rx + lx * hw, rx - lx * hw]
        ys = [fy + ly * hw, fy - ly * hw, ry + ly * hw, ry - ly * hw]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class CompletedTrip:
    vid: int
    approach: str
    kind: str
    wait_s: float
    travel_s: float


class JunctionWorld:
    def __init__(self, arm_len: float = 80.0, arrivals_per_min: Optional[dict] = None,
                 vehicle_mix: Optional[dict] = None, speed_limit_kmph: float = 50.0,
                 seed: int = 42, emergency_per_hour: float = 0.0) -> None:
        self.arm_len = arm_len
        self.rng = random.Random(seed)               # arrivals only
        # Vehicle attributes get their own stream: anything that delays a spawn
        # (an injected vehicle at the entry) must not shift the arrival sequence.
        self.veh_rng = random.Random(seed + 2)
        # Separate generator: switching random ambulances on/off leaves normal traffic identical.
        self.emergency_rng = random.Random(seed + 1)
        self.emergency_per_hour = emergency_per_hour
        self.arrivals = dict(arrivals_per_min or {"N": 10, "E": 10, "S": 10, "W": 10})
        self.mix = dict(vehicle_mix or {"car": 1.0})
        self.speed_limit = speed_limit_kmph / 3.6
        self.t = 0.0
        self.vehicles: list[SimVehicle] = []
        self.completed: list[CompletedTrip] = []
        self.crashes: list[dict] = []          # ground truth, for testing detectors
        self._next_id = 1
        self._pending: dict[str, int] = {a: 0 for a in APPROACH_DIR}   # vehicles waiting to enter
        self.signal: dict[str, str] = {a: "RED" for a in APPROACH_DIR}
        self.s_stop = arm_len - BOX_HALF - STOP_GAP        # stop line position along path
        self.s_box_in = arm_len - BOX_HALF
        self.s_box_out = arm_len + BOX_HALF

    # ------------------------------------------------------------ spawning
    def _new_vehicle(self, approach: str, kind: Optional[str] = None, rng: Optional[random.Random] = None,
                     **flags) -> SimVehicle:
        rng = rng or self.veh_rng
        if kind is None:
            kinds, weights = zip(*self.mix.items())
            kind = rng.choices(kinds, weights)[0]
        length, width, vf = KIND_SPECS[kind]
        v0 = self.speed_limit * vf * rng.uniform(0.8, 1.0)
        color = (235, 235, 235) if kind == "ambulance" else rng.choice(CAR_COLORS)
        veh = SimVehicle(
            vid=self._next_id, kind=kind, approach=approach, s=0.0, v=v0 * 0.8, v0=v0,
            length=length, width=width, color=color, plate=random_plate(rng),
            t_spawn=self.t,
            helmet=(rng.random() > 0.1) if kind == "motorcycle" else None,
        )
        for k, val in flags.items():
            setattr(veh, k, val)
        if veh.siren and kind == "ambulance":
            veh.v0 = self.speed_limit * 1.1
        if veh.speeding:
            veh.v0 = veh.v = self.speed_limit * 1.5
        if veh.stalled:
            veh.v = 0.0
        self._next_id += 1
        return veh

    def _entry_clear(self, approach: str, wrong_way: bool = False) -> bool:
        for v in self.vehicles:
            if v.approach == approach and v.wrong_way == wrong_way and v.s - v.length < 8.0:
                return False
        return True

    def inject(self, approach: str, kind: Optional[str] = None, rng: Optional[random.Random] = None,
               **flags) -> SimVehicle:
        """Add a special vehicle now (used by the dashboard demo buttons and tests)."""
        veh = self._new_vehicle(approach, kind, rng, **flags)
        if veh.wrong_way:
            veh.s = 0.0
        else:
            # Start behind the last vehicle if the entry is busy, no faster than it.
            same = [v for v in self.vehicles if v.approach == approach and not v.wrong_way]
            if same:
                last = min(same, key=lambda v: v.s - v.length)
                veh.s = min(0.0, last.s - last.length - S0 - 1.0)
                if last.s - last.length < 40.0:
                    veh.v = min(veh.v, last.v)
            else:
                veh.s = 0.0
        self.vehicles.append(veh)
        return veh

    # ------------------------------------------------------------ crash demo
    VIEW_REACH = 40.0   # metres before the junction centre that every approach camera sees

    def _uncrossed(self, approach: str) -> list[SimVehicle]:
        return [v for v in self.vehicles if v.approach == approach and not v.wrong_way and not v.crossed_stop]

    def crash_score(self, approach: str) -> int:
        """2 = a stopped queue whose tail is on camera, 1 = empty lane, 0 = moving traffic."""
        lane = self._uncrossed(approach)
        if not lane:
            return 1
        rear = min(lane, key=lambda v: v.s)
        on_camera = rear.s - rear.length >= self.arm_len - self.VIEW_REACH
        return 2 if (rear.v < 1.0 and on_camera) else 0

    SPEEDER_START = 10.0    # metres before the camera's view where the demo speeder appears
    SPEEDER_ROOM = 35.0     # clear road it needs ahead to be measured at full speed

    def speeder_room(self, approach: str) -> bool:
        start = self.arm_len - self.VIEW_REACH - self.SPEEDER_START
        return all(v.s - v.length - start >= self.SPEEDER_ROOM or v.s < start - 8.0
                   for v in self.vehicles if v.approach == approach and not v.wrong_way)

    def inject_speeder(self, approach: str) -> SimVehicle:
        """Demo: a speeder at full speed just before the camera's view (when the
        road ahead is clear), so it is measured before it has to brake."""
        if not self.speeder_room(approach):
            return self.inject(approach, "car", speeding=True)
        veh = self._new_vehicle(approach, "car", speeding=True)
        veh.s = self.arm_len - self.VIEW_REACH - self.SPEEDER_START
        self.vehicles.append(veh)
        return veh

    def inject_crash(self, approach: str) -> SimVehicle:
        """Demo: a distracted driver rear-ends the stopped vehicle at the back of the
        queue, in view of the camera. With no queue, a car breaks down first."""
        lane = self._uncrossed(approach)
        if lane:
            target = min(lane, key=lambda v: v.s)
        else:
            target = self.inject(approach, "car", stalled=True)
            target.s = self.s_stop - 15.0
        car = self._new_vehicle(approach, "car", distracted=True)
        car.s = max(0.0, target.s - target.length - 30.0)
        car.v = car.v0
        self.vehicles.append(car)
        return car

    # ------------------------------------------------------------ dynamics
    def _leader_gap(self, veh: SimVehicle) -> tuple[float, float]:
        """Gap (m) to the nearest obstacle ahead and its speed. inf if none."""
        gap, lead_v = math.inf, 0.0
        if not veh.distracted:
            for o in self.vehicles:
                if o is veh or o.approach != veh.approach or o.wrong_way != veh.wrong_way:
                    continue
                if o.s > veh.s:
                    g = o.s - o.length - veh.s
                    if g < gap:
                        gap, lead_v = g, o.v
            # Crashed vehicles inside the junction box block everyone crossing it.
            if not veh.crossed_stop or veh.s < self.s_box_out:
                for o in self.vehicles:
                    if o.crashed and o.approach != veh.approach:
                        x0, y0, x1, y1 = o.footprint(self.arm_len)
                        if x1 > -BOX_HALF and x0 < BOX_HALF and y1 > -BOX_HALF and y0 < BOX_HALF:
                            g = self.s_stop - veh.s if veh.s < self.s_stop else self.s_box_in - veh.s
                            if 0 <= g < gap:
                                gap, lead_v = g, 0.0
        # Red / amber stop line (wrong-way vehicles stop at the box edge)
        if veh.wrong_way:
            g = self.s_box_in - 1.0 - veh.s
            if g < gap:
                gap, lead_v = g, 0.0
        elif not veh.crossed_stop and not veh.ignore_red:
            to_line = self.s_stop - veh.s
            light = self.signal[veh.approach]
            must_stop = light == "RED" or (light == "AMBER" and to_line > veh.v ** 2 / (2 * B_COMF))
            if not must_stop and self._box_blocked(veh):
                must_stop = True            # don't enter while cross traffic is still in the box
            if must_stop and to_line < gap:
                gap, lead_v = to_line, 0.0
        return gap, lead_v

    def _box_blocked(self, veh: SimVehicle) -> bool:
        my_axis = veh.dir[0] == 0
        for o in self.vehicles:
            if o.wrong_way or (o.dir[0] == 0) == my_axis:
                continue
            if self.s_box_in - 1 < o.s < self.s_box_out + o.length:
                return True
        return False

    def _idm_acc(self, veh: SimVehicle, gap: float, lead_v: float) -> float:
        # Speeders are aggressive: hard acceleration, short headway, late hard braking.
        a_max, head, b = (2 * A_MAX, 0.6, 5.0) if veh.speeding else (A_MAX, T_HEAD, B_COMF)
        free = 1 - (veh.v / max(veh.v0, 0.1)) ** 4
        if math.isinf(gap):
            return a_max * free
        s_star = S0 + max(0.0, veh.v * head + veh.v * (veh.v - lead_v) / (2 * math.sqrt(a_max * b)))
        return a_max * (free - (s_star / max(gap, 0.1)) ** 2)

    def step(self, dt: float) -> list[dict]:
        """Advance the world by dt seconds. Returns newly happened ground-truth
        incidents (crashes) so tests can check our detectors."""
        self.t += dt
        # 1) arrivals (Poisson process per approach)
        for a, rate in self.arrivals.items():
            lam = rate / 60.0 * dt
            if self.rng.random() < lam:
                self._pending[a] += 1
            if self._pending[a] and self._entry_clear(a):
                self._pending[a] -= 1
                self.vehicles.append(self._new_vehicle(a))
        # 1b) random emergency vehicles (whole junction, random approach)
        if self.emergency_per_hour > 0 and self.emergency_rng.random() < self.emergency_per_hour / 3600.0 * dt:
            self.inject(self.emergency_rng.choice(list(self.arrivals)), "ambulance",
                        rng=self.emergency_rng, siren=True)
        # 2) move vehicles
        for veh in self.vehicles:
            if veh.crashed or veh.stalled:
                veh.v = 0.0
                continue
            gap, lead_v = self._leader_gap(veh)
            acc = max(-9.0, self._idm_acc(veh, gap, lead_v))    # -9 m/s^2 = emergency braking limit
            veh.v = max(0.0, veh.v + acc * dt)
            veh.s += veh.v * dt
            if not veh.crossed_stop and veh.v < 0.5 and veh.s < self.s_stop:
                veh.wait_s += dt
            if not veh.crossed_stop and veh.s >= self.s_stop:
                veh.crossed_stop = True
            end = self.s_box_in if veh.wrong_way else 2 * self.arm_len + veh.length
            if veh.s >= end and not veh.wrong_way:
                veh.done = True
                self.completed.append(CompletedTrip(veh.vid, veh.approach, veh.kind, veh.wait_s,
                                                    self.t - veh.t_spawn))
            if veh.wrong_way and veh.v < 0.1 and veh.s > self.s_box_in - 5:
                veh.done = True   # wrong-way driver turns away at the junction
        # 3) collisions (axis-aligned footprint overlap)
        new_crashes = self._check_collisions()
        # 4) clear crashed vehicles after 120 s (tow truck), remove finished
        for veh in self.vehicles:
            if veh.crashed and veh.crash_t is not None and self.t - veh.crash_t > 120:
                veh.done = True
            elif veh.stalled and not veh.crashed and self.t - veh.t_spawn > 60:
                veh.done = True                     # towed away
        self.vehicles = [v for v in self.vehicles if not v.done]
        return new_crashes

    def _check_collisions(self) -> list[dict]:
        events = []
        boxes = [(v, v.footprint(self.arm_len)) for v in self.vehicles]
        for i in range(len(boxes)):
            a, (ax0, ay0, ax1, ay1) = boxes[i]
            for j in range(i + 1, len(boxes)):
                b, (bx0, by0, bx1, by1) = boxes[j]
                if (a.crashed and b.crashed) or a.wrong_way or b.wrong_way:
                    continue   # wrong-way drivers are a violation demo, not a crash demo
                if ax1 > bx0 and bx1 > ax0 and ay1 > by0 and by1 > ay0:
                    for v in (a, b):
                        if not v.crashed:
                            v.crashed, v.crash_t, v.v = True, self.t, 0.0
                    cx, cy = (max(ax0, bx0) + min(ax1, bx1)) / 2, (max(ay0, by0) + min(ay1, by1)) / 2
                    ev = {"t": self.t, "vids": (a.vid, b.vid), "xy": (cx, cy),
                          "approaches": (a.approach, b.approach)}
                    self.crashes.append(ev)
                    events.append(ev)
        return events

    # ------------------------------------------------------------ stats
    def stats(self, since: float = 0.0) -> dict:
        trips = [c for c in self.completed if self.t - c.travel_s >= since]
        n = len(trips)
        return {
            "t": round(self.t, 1),
            "completed": n,
            "avg_wait_s": round(sum(c.wait_s for c in trips) / n, 2) if n else 0.0,
            "max_wait_s": round(max((c.wait_s for c in trips), default=0.0), 1),
            "on_road": len(self.vehicles),
            "pending_entry": sum(self._pending.values()),
        }
