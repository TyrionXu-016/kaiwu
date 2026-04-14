#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import numpy as np
from collections import deque


def _coerce_env_reward(x):
    """Normalize env.step first return to scalar for reward blending."""
    if x is None:
        return 0.0
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return 0.0
            return float(np.asarray(x).reshape(-1)[0])
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径

    # Auxiliary shaping (keep smaller than cleaning so RL optimizes "tiles cleaned" first).
    # 辅助塑形；须弱于清扫项，避免与官方「得分=清扫地面数」错位。
    APPROACH_DIRT_REWARD = 0.004

    CHARGE_GAIN_COEF = 0.006

    APPROACH_CHARGER_REWARD = 0.008
    LOW_BATTERY_RATIO = 0.42
    HARD_GUARD_BATTERY_RATIO = 0.30
    # Absolute guard threshold: when battery <= this value, force go charge.
    # 绝对电量阈值：当电量低于该值时，硬保护强制回充。
    HARD_GUARD_BATTERY_ABS = 150
    # Runtime safety margin for "battery vs nearest charger distance" constraint.
    # 运行时安全余量：用于约束“电量必须覆盖最近充电桩距离”。
    CHARGE_SAFETY_MARGIN = 2.0
    # Prefer cardinal moves for coverage pattern (0/2/4/6) in non-charging mode.
    # 非回充模式下优先上下左右，减少斜线清扫。
    ENABLE_CARDINAL_CLEAN_BIAS = False
    ENABLE_FRONTIER_PLANNER = False
    ENABLE_REVISIT_PENALTY = False
    FRONTIER_RESELECT_INTERVAL = 14
    REVISIT_PENALTY_COEF = -0.0008
    REVISIT_PENALTY_CAP = 6
    CHARGE_MODE_EXIT_BATTERY = 260
    # NPC safety cost for frontier planning.
    # Frontier 选点和落脚动作都会计入 NPC 安全代价（切比雪夫距离）。
    NPC_DANGER_RADIUS = 1
    NPC_CAUTION_RADIUS = 4
    NPC_DANGER_PENALTY = 1200.0
    NPC_CAUTION_COEF = 42.0
    GUARD_PROGRESS_EPS = 0.15
    GUARD_STUCK_STEPS = 6
    GUARD_REVISIT_COEF = 0.9
    CHARGE_STRICT_MARGIN = 1.5
    GUARD_NPC_DANGER_RADIUS = 2
    CHARGER_SWITCH_STUCK_STEPS = 10
    CHARGE_BFS_MAX_EXPAND = 5000
    GUARD_RELAX_STUCK_STEPS = 8
    CRITICAL_BATTERY_RATIO = 0.22
    LOW_BATTERY_STEP_PENALTY = -0.0025
    CRITICAL_BATTERY_STEP_PENALTY = -0.005

    # Primary signal: matches official score direction (more cleaned tiles -> higher reward).
    # 主信号：与「清扫地面数量」一致，权重大于各类塑形。
    CLEANING_TILE_WEIGHT = 0.18

    STEP_PENALTY_IDLE = -0.0012
    STEP_PENALTY_ACTIVE = -0.0004

    # Blend official env_reward (per-step task score delta) into RL reward; set 0 if it double-counts with cleaning.
    # 将环境返回的 env_reward（与任务得分相关）混入训练回报；若与 dirt_cleaned 重复可改为 0。
    ENV_REWARD_BLEND = 0.12

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self.nearest_charger_dist = 200.0
        self.last_nearest_charger_dist = 200.0

        self._prev_battery = 0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        self._frame_env_reward = 0.0
        self.charger_cells = []
        self.charger_groups = {}
        self.charge_guard_triggered = 0
        self.charge_guard_trigger_count = 0
        self.memory_map = np.full((self.GRID_SIZE, self.GRID_SIZE), -1, dtype=np.int8)  # -1 unknown, 0 obstacle, 1 clean, 2 dirty
        self.visit_count = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int16)
        self.frontier_target = None
        self.last_frontier_reselect_step = -9999
        self.in_charge_mode = False
        self.npc_positions = []
        self._guard_prev_dist = 200.0
        self._guard_no_progress_steps = 0
        self._target_charger_id = None

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]
        organs = frame_state.get("organs") or []
        npcs = frame_state.get("npcs") or []

        prev_battery = self.battery

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Battery / 电量
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)
        self._prev_battery = prev_battery

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)
            self._update_memory_map(hx, hz)
        x, z = self.cur_pos
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            self.visit_count[x, z] = min(30000, int(self.visit_count[x, z]) + 1)
        self._update_chargers(organs)
        self._update_npcs(npcs)

    def _update_npcs(self, npcs):
        """Cache NPC global positions from frame_state."""
        pos = []
        for npc in npcs:
            p = npc.get("pos") or {}
            x = int(p.get("x", -1))
            z = int(p.get("z", -1))
            if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
                pos.append((x, z))
        self.npc_positions = pos

    def _npc_safety_cost(self, x, z):
        """Safety cost at grid (x,z) based on nearest NPC.

        Uses Chebyshev distance to align with 3x3 collision neighborhood.
        """
        if not self.npc_positions:
            return 0.0
        min_d = min(max(abs(x - nx), abs(z - nz)) for nx, nz in self.npc_positions)
        if min_d <= self.NPC_DANGER_RADIUS:
            return self.NPC_DANGER_PENALTY
        if min_d <= self.NPC_CAUTION_RADIUS:
            # Distance-aware caution zone, closer => larger cost.
            return self.NPC_CAUTION_COEF * float(self.NPC_CAUTION_RADIUS - min_d + 1)
        return 0.0

    def _is_npc_danger_cell(self, x, z, radius=None):
        """Whether cell is inside NPC collision danger area (Chebyshev <= danger radius)."""
        if radius is None:
            radius = self.NPC_DANGER_RADIUS
        if not self.npc_positions:
            return False
        for nx, nz in self.npc_positions:
            if max(abs(x - nx), abs(z - nz)) <= radius:
                return True
        return False

    def _update_memory_map(self, hx, hz):
        """Project 21x21 local map into global memory map."""
        view = self._view_map
        if view is None:
            return
        vsize = view.shape[0]
        half = vsize // 2
        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if not (0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE):
                    continue
                cell = int(view[ri, ci])
                if cell == 0:
                    self.memory_map[gx, gz] = 0
                elif cell == 1:
                    self.memory_map[gx, gz] = 1
                elif cell == 2:
                    self.memory_map[gx, gz] = 2

    def _update_chargers(self, organs):
        """Parse charger cells from frame_state.organs.

        通过 organs 解析充电桩占据的格子集合（支持 3x3 或其他尺寸）。
        """
        cells = []
        groups = {}
        for organ in organs:
            # sub_type=1 means charger in official protocol.
            if int(organ.get("sub_type", 0)) != 1:
                continue
            pos = organ.get("pos") or {}
            ox = int(pos.get("x", -1))
            oz = int(pos.get("z", -1))
            w = max(1, int(organ.get("w", 1)))
            h = max(1, int(organ.get("h", 1)))
            gid = int(organ.get("config_id", len(groups)))
            if gid not in groups:
                groups[gid] = []
            for dx in range(w):
                for dz in range(h):
                    cx = ox + dx
                    cz = oz + dz
                    if 0 <= cx < self.GRID_SIZE and 0 <= cz < self.GRID_SIZE:
                        cells.append((cx, cz))
                        groups[gid].append((cx, cz))
        # Fallback: some env variants may not expose organs every frame,
        # but chargers may still be encoded in local map_info as 3/4.
        # 兜底：部分环境帧不返回 organs，但 map_info 里可能仍有充电桩编码（3/4）。
        if not cells and self._view_map is not None:
            hx, hz = self.cur_pos
            center = self.VIEW_HALF
            coords = np.argwhere((self._view_map == 3.0) | (self._view_map == 4.0))
            if -1 not in groups:
                groups[-1] = []
            for rx, rz in coords:
                gx = hx + int(rx) - center
                gz = hz + int(rz) - center
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    cells.append((gx, gz))
                    groups[-1].append((gx, gz))

        # De-duplicate to keep distance computation stable.
        # 去重，避免重复点影响后续距离计算效率。
        if cells:
            cells = list(dict.fromkeys(cells))
            for gid, pts in list(groups.items()):
                groups[gid] = list(dict.fromkeys(pts))
        self.charger_cells = cells
        self.charger_groups = groups

    def _select_charger_group_points(self, force_switch=False):
        """Select active charger group (nearest or second-nearest when stuck)."""
        if not self.charger_groups:
            return []
        hx, hz = self.cur_pos
        ranked = []
        for gid, pts in self.charger_groups.items():
            if not pts:
                continue
            arr = np.array(pts, dtype=np.float32)
            d = float(np.min(np.sqrt((arr[:, 0] - hx) ** 2 + (arr[:, 1] - hz) ** 2)))
            ranked.append((d, gid))
        if not ranked:
            return []
        ranked.sort(key=lambda x: x[0])
        pick_gid = ranked[0][1]
        if force_switch and len(ranked) > 1:
            # If stuck, switch to next nearest charger to avoid dead-end looping.
            # 卡住时切换次近充电桩，避免围绕单桩死循环。
            pick_gid = ranked[1][1]
        self._target_charger_id = pick_gid
        return self.charger_groups.get(pick_gid, [])

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    self.passable_map[gx, gz] = 1 if view[ri, ci] != 0 else 0

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21.

        局部视野特征（49D）：从 21×21 视野中心裁剪 7×7。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        """Global state feature (14D).

        全局状态特征（14D）：在 12D 基础上增加充电桩距离与是否接近充电桩。

        Dimensions / 维度说明：
          [0]-[11] 同原 12 维（步数、电量、清扫、位置、四向污渍射线、最近污渍、接近污渍）
          [12] nearest_charger_norm 视野内最近充电桩格到自身的距离归一化
          [13] charger_delta        低电量时是否在接近充电桩（1/0，阈值见 LOW_BATTERY_RATIO）
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(x - (hx - self.VIEW_HALF), 0, 20), np.clip(z - (hz - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        self.last_nearest_charger_dist = self.nearest_charger_dist
        self.nearest_charger_dist = self._calc_nearest_charger_dist()
        nearest_charger_norm = _norm(self.nearest_charger_dist, 180.0)
        charger_delta = 0.0
        if battery_ratio < self.LOW_BATTERY_RATIO and self.nearest_charger_dist < self.last_nearest_charger_dist:
            charger_delta = 1.0

        return np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
                nearest_charger_norm,
                charger_delta,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def _calc_nearest_charger_dist(self):
        """Nearest Euclidean distance to charger using global organ positions first."""
        if self.charger_cells:
            hx, hz = self.cur_pos
            pts = np.array(self.charger_cells, dtype=np.float32)
            dists = np.sqrt((pts[:, 0] - hx) ** 2 + (pts[:, 1] - hz) ** 2)
            return float(np.min(dists))

        # Fallback for envs that encode charger values directly in map_info.
        view = self._view_map
        if view is None:
            return 200.0
        coords = np.argwhere((view == 3.0) | (view == 4.0))
        if len(coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((coords[:, 0] - center) ** 2 + (coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def _bfs_steps_to_charger(self, start, charger_set, valid_move_func):
        """Shortest path steps from start to any charger; returns None if unreachable."""
        if start in charger_set:
            return 0
        q = deque([start])
        visited = {start}
        depth = {start: 0}
        dirs = [
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
        expand = 0
        while q:
            x, z = q.popleft()
            d0 = depth[(x, z)]
            expand += 1
            if expand > self.CHARGE_BFS_MAX_EXPAND:
                return None
            for dx, dz in dirs:
                nx, nz = x + dx, z + dz
                if (nx, nz) in visited:
                    continue
                if not valid_move_func(x, z, dx, dz):
                    continue
                if (nx, nz) in charger_set:
                    return d0 + 1
                visited.add((nx, nz))
                depth[(nx, nz)] = d0 + 1
                q.append((nx, nz))
        return None

    def get_charge_guard_action(self, legal_action, last_action=-1):
        """Hard safety guard: force action toward nearest charger under low battery.

        低电量硬保护：当电量低于阈值时，优先选择能让自身更靠近最近充电桩的合法动作。
        返回动作索引（0-7），若无需保护则返回 None。
        """
        self.charge_guard_triggered = 0
        if not self.charger_cells:
            return None

        hx, hz = self.cur_pos
        use_switch = self._guard_no_progress_steps >= self.CHARGER_SWITCH_STUCK_STEPS
        active_pts = self._select_charger_group_points(force_switch=use_switch)
        if not active_pts:
            active_pts = self.charger_cells
        charger_pts = np.array(active_pts, dtype=np.float32)
        base_dist = float(np.min(np.sqrt((charger_pts[:, 0] - hx) ** 2 + (charger_pts[:, 1] - hz) ** 2)))
        if base_dist <= 0.5:
            return None

        br = float(self.battery) / float(max(self.battery_max, 1))
        must_charge = (
            self.battery <= self.HARD_GUARD_BATTERY_ABS
            or br < self.HARD_GUARD_BATTERY_RATIO
            or float(self.battery) <= (base_dist + self.CHARGE_SAFETY_MARGIN)
        )
        if not must_charge:
            self._guard_no_progress_steps = 0
            self._guard_prev_dist = base_dist
            return None

        if base_dist < (self._guard_prev_dist - self.GUARD_PROGRESS_EPS):
            self._guard_no_progress_steps = 0
        else:
            self._guard_no_progress_steps += 1
        self._guard_prev_dist = base_dist

        dirs = [
            (1, 0),    # 0 右
            (1, -1),   # 1 右上
            (0, -1),   # 2 上
            (-1, -1),  # 3 左上
            (-1, 0),   # 4 左
            (-1, 1),   # 5 左下
            (0, 1),    # 6 下
            (1, 1),    # 7 右下
        ]

        def passable(x, z):
            if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                return False
            return bool(self.passable_map[x, z] == 1)

        def valid_move(x, z, dx, dz, danger_radius=None):
            if danger_radius is None:
                danger_radius = self.GUARD_NPC_DANGER_RADIUS
            tx, tz = x + dx, z + dz
            if not passable(tx, tz):
                return False
            if self._is_npc_danger_cell(tx, tz, radius=danger_radius):
                return False
            if dx != 0 and dz != 0:
                # Diagonal anti-corner rule: at least one side neighbor passable.
                # 斜向防穿角：水平/垂直至少一侧可通行。
                side_ok = passable(x + dx, z) or passable(x, z + dz)
                if not side_ok:
                    return False
                if self._is_npc_danger_cell(
                    x + dx, z, radius=danger_radius
                ) and self._is_npc_danger_cell(x, z + dz, radius=danger_radius):
                    return False
                return True
            return True

        opposite = {0: 4, 4: 0, 2: 6, 6: 2, 1: 5, 5: 1, 3: 7, 7: 3}

        # Build local BFS target set: all visible/known charger cells.
        # 构建 BFS 目标集合：已知充电桩格子。
        charger_set = set((int(cx), int(cz)) for cx, cz in self.charger_cells)
        start = (hx, hz)

        def bfs_next_actions():
            """Return set of first-step actions on shortest paths to charger."""
            if start in charger_set:
                return set()
            q = deque([start])
            visited = {start}
            parent = {}
            found_targets = []
            depth = {start: 0}
            min_depth = None
            while q:
                x, z = q.popleft()
                d0 = depth[(x, z)]
                if min_depth is not None and d0 > min_depth:
                    break
                if (x, z) in charger_set:
                    found_targets.append((x, z))
                    min_depth = d0
                    continue
                for a, (dx, dz) in enumerate(dirs):
                    nx, nz = x + dx, z + dz
                    if (nx, nz) in visited:
                        continue
                    if not valid_move(x, z, dx, dz):
                        continue
                    visited.add((nx, nz))
                    parent[(nx, nz)] = ((x, z), a)
                    depth[(nx, nz)] = d0 + 1
                    q.append((nx, nz))
            if not found_targets:
                return set()
            next_actions = set()
            for tgt in found_targets:
                cur = tgt
                first_a = None
                while cur in parent:
                    prev, a = parent[cur]
                    first_a = a
                    cur = prev
                    if cur == start:
                        break
                if first_a is not None:
                    next_actions.add(first_a)
            return next_actions

        bfs_actions = bfs_next_actions()

        relax_radius = self.GUARD_NPC_DANGER_RADIUS
        if self._guard_no_progress_steps >= self.GUARD_RELAX_STUCK_STEPS:
            relax_radius = max(0, self.GUARD_NPC_DANGER_RADIUS - 1)

        best_action = None
        best_score = float("inf")
        best_dist = float("inf")
        best_safe_action = None
        best_safe_score = float("inf")
        best_margin_action = None
        best_margin = -1e9
        fallback_action = None
        fallback_score = float("inf")
        for a, (dx, dz) in enumerate(dirs):
            if a >= len(legal_action) or int(legal_action[a]) != 1:
                continue
            if not valid_move(hx, hz, dx, dz, danger_radius=relax_radius):
                continue
            nx, nz = hx + dx, hz + dz
            euclid_d = float(np.min(np.sqrt((charger_pts[:, 0] - nx) ** 2 + (charger_pts[:, 1] - nz) ** 2)))
            bfs_steps = self._bfs_steps_to_charger((nx, nz), charger_set, valid_move)
            if bfs_steps is None:
                d = euclid_d + 1000.0
            else:
                d = float(bfs_steps)
            npc_cost = self._npc_safety_cost(nx, nz)
            revisit_cost = self.GUARD_REVISIT_COEF * float(min(8, int(self.visit_count[nx, nz])))
            reverse_penalty = 0.25 if opposite.get(last_action, -1) == a else 0.0
            bfs_bonus = -2.0 if (a in bfs_actions) else 0.0
            score = d + 0.55 * npc_cost + revisit_cost + reverse_penalty + bfs_bonus

            if score < best_score:
                best_score = score
                best_dist = d
                best_action = a
            if score < fallback_score:
                fallback_score = score
                fallback_action = a
            # Strict runtime safety: after this move, remaining battery must still cover nearest charger distance.
            # 严格运行时约束：执行该步后剩余电量仍需覆盖最近充电桩距离。
            remain_after_step = float(self.battery - 1)
            strict_margin = remain_after_step - (d + self.CHARGE_STRICT_MARGIN)
            if strict_margin > best_margin:
                best_margin = strict_margin
                best_margin_action = a
            if strict_margin >= 0.0:
                if score < best_safe_score:
                    best_safe_score = score
                    best_safe_action = a

        # Priority 1: choose safest action that keeps "battery >= charger distance" invariant.
        # 优先级1：选择满足“电量>=最近充电桩距离”不变式的动作。
        if best_safe_action is not None:
            self.charge_guard_triggered = 1
            self.charge_guard_trigger_count += 1
            return int(best_safe_action)
        # If strict safe action doesn't exist, allow near-feasible action as emergency fallback.
        # 当无严格可达动作时，仅在“接近可达”情况下放宽一步，避免完全失控。
        if best_margin_action is not None and best_margin >= -0.5:
            self.charge_guard_triggered = 1
            self.charge_guard_trigger_count += 1
            return int(best_margin_action)
        # If stuck for several steps, allow escape action that trades a bit of distance
        # for much safer / less repeated cells, to bypass local minima near walls.
        # 若连续多步无进展，则允许“绕障逃逸”动作，避免在墙角原地消耗电量。
        if self._guard_no_progress_steps >= self.GUARD_STUCK_STEPS and best_action is not None:
            self.charge_guard_triggered = 1
            self.charge_guard_trigger_count += 1
            return int(best_action)
        # Priority 2 (degraded): no fully safe move available, still force nearest-charger action.
        # 优先级2（退化保护）：若无完全安全动作，仍强制选最接近充电桩的动作，尽量自救。
        if best_action is not None and best_dist < base_dist:
            self.charge_guard_triggered = 1
            self.charge_guard_trigger_count += 1
            return int(best_action)
        if fallback_action is not None:
            self.charge_guard_triggered = 1
            self.charge_guard_trigger_count += 1
            return int(fallback_action)
        return None

    def get_cardinal_clean_action(self, legal_action, probs, last_action):
        """Prefer axis-aligned movement for map-tiling style cleaning.

        平铺清扫偏置：优先选择上下左右动作，降低斜向走位占比。
        返回动作索引（0-7），若不接管则返回 None。
        """
        if not self.ENABLE_CARDINAL_CLEAN_BIAS:
            return None
        if not self.charger_cells:
            # No charger info is okay, but keep behavior conservative if perception is weak.
            # 无充电桩信息时仍可运行；这里只不做额外限制。
            pass

        br = float(self.battery) / float(max(self.battery_max, 1))
        # Do not apply cleaning bias when charging guard may be needed.
        # 回充优先级更高：低电时不做平铺偏置接管。
        if self.battery <= self.HARD_GUARD_BATTERY_ABS or br < self.HARD_GUARD_BATTERY_RATIO:
            return None

        hx, hz = self.cur_pos
        cardinals = [0, 2, 4, 6]  # right, up, left, down
        dirs = {
            0: (1, 0),
            2: (0, -1),
            4: (-1, 0),
            6: (0, 1),
        }
        opposite = {0: 4, 4: 0, 2: 6, 6: 2}

        def passable(x, z):
            if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                return False
            return bool(self.passable_map[x, z] == 1)

        best_a = None
        best_score = -1e9
        for a in cardinals:
            if a >= len(legal_action) or int(legal_action[a]) != 1:
                continue
            dx, dz = dirs[a]
            tx, tz = hx + dx, hz + dz
            if not passable(tx, tz):
                continue
            score = float(probs[a]) if a < len(probs) else 0.0
            # Keep moving straight to form longer lanes.
            # 保持同向，形成更长的平铺条带。
            if last_action == a:
                score += 0.05
            if opposite.get(last_action, -1) == a:
                score -= 0.03
            if score > best_score:
                best_score = score
                best_a = a

        return None if best_a is None else int(best_a)

    def _in_charge_mode_now(self):
        """Charge mode switch with simple hysteresis."""
        need_charge = (
            self.battery <= self.HARD_GUARD_BATTERY_ABS
            or self.battery <= (self.nearest_charger_dist + self.CHARGE_SAFETY_MARGIN)
        )
        if need_charge:
            self.in_charge_mode = True
        elif self.battery >= self.CHARGE_MODE_EXIT_BATTERY:
            self.in_charge_mode = False
        return self.in_charge_mode

    def _reselect_frontier_target(self):
        """Pick a frontier-like target from global memory.

        Priority:
        1) Dirty cells (value=2) near current position.
        2) Clean cells adjacent to unknown area (exploration frontier).
        """
        hx, hz = self.cur_pos
        dirty = np.argwhere(self.memory_map == 2)
        if len(dirty) > 0:
            # Prefer closer dirty tiles; slight bonus for less-visited areas.
            best = None
            best_score = 1e18
            for x, z in dirty:
                x = int(x)
                z = int(z)
                dist_score = float((x - hx) ** 2 + (z - hz) ** 2)
                visit_bias = 0.12 * float(self.visit_count[x, z])
                npc_cost = self._npc_safety_cost(x, z)
                score = dist_score + visit_bias + npc_cost
                if score < best_score:
                    best_score = score
                    best = (x, z)
            self.frontier_target = best
            self.last_frontier_reselect_step = self.step_no
            return

        clean = np.argwhere(self.memory_map == 1)
        best = None
        best_score = 1e18
        for x, z in clean:
            x = int(x)
            z = int(z)
            has_unknown_neighbor = False
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, nz = x + dx, z + dz
                if 0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE and self.memory_map[nx, nz] == -1:
                    has_unknown_neighbor = True
                    break
            if not has_unknown_neighbor:
                continue
            score = (
                float((x - hx) ** 2 + (z - hz) ** 2)
                + 0.1 * float(self.visit_count[x, z])
                + self._npc_safety_cost(x, z)
            )
            if score < best_score:
                best_score = score
                best = (x, z)
        self.frontier_target = best
        self.last_frontier_reselect_step = self.step_no

    def get_frontier_action(self, legal_action, last_action):
        """Coverage planner action toward frontier target.

        Returns action index or None.
        """
        if not self.ENABLE_FRONTIER_PLANNER:
            return None
        if self._in_charge_mode_now():
            # Pause coverage task in low-battery mode.
            # 低电量切换回充模式，暂停覆盖目标。
            self.frontier_target = None
            return None

        hx, hz = self.cur_pos
        need_reselect = (
            self.frontier_target is None
            or (self.step_no - self.last_frontier_reselect_step) >= self.FRONTIER_RESELECT_INTERVAL
        )
        if self.frontier_target is not None:
            tx, tz = self.frontier_target
            if (hx, hz) == (tx, tz):
                need_reselect = True
            elif 0 <= tx < self.GRID_SIZE and 0 <= tz < self.GRID_SIZE and self.memory_map[tx, tz] != 2:
                # If target no longer dirty, reselect.
                need_reselect = True
        if need_reselect:
            self._reselect_frontier_target()
        if self.frontier_target is None:
            return None

        tx, tz = self.frontier_target
        dirs = [
            (1, 0),    # 0 right
            (1, -1),   # 1 up-right
            (0, -1),   # 2 up
            (-1, -1),  # 3 up-left
            (-1, 0),   # 4 left
            (-1, 1),   # 5 down-left
            (0, 1),    # 6 down
            (1, 1),    # 7 down-right
        ]

        def passable(x, z):
            if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                return False
            return bool(self.passable_map[x, z] == 1)

        def valid_move(x, z, dx, dz):
            nx, nz = x + dx, z + dz
            if not passable(nx, nz):
                return False
            if dx != 0 and dz != 0:
                return passable(x + dx, z) or passable(x, z + dz)
            return True

        best_action = None
        best_score = 1e18
        for a, (dx, dz) in enumerate(dirs):
            if a >= len(legal_action) or int(legal_action[a]) != 1:
                continue
            if not valid_move(hx, hz, dx, dz):
                continue
            nx, nz = hx + dx, hz + dz
            # Distance-to-target primary objective.
            score = float((nx - tx) ** 2 + (nz - tz) ** 2)
            # One-step NPC safety lookahead.
            score += self._npc_safety_cost(nx, nz)
            # Mild lane consistency for smoother sweeping.
            if last_action == a:
                score -= 0.15
            if score < best_score:
                best_score = score
                best_action = a
        return None if best_action is None else int(best_action)

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)

    def feature_process(self, env_obs, last_action, step_env_reward=0.0):
        """Generate 71D feature vector, legal action mask, and scalar reward.

        生成 71D 特征向量、合法动作掩码和标量奖励。
        step_env_reward: first return value of env.step (official score signal for this transition).
        """
        self.pb2struct(env_obs, last_action)
        self._frame_env_reward = _coerce_env_reward(step_env_reward)

        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()  # 14D
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr])  # 71D

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_reward = self.CLEANING_TILE_WEIGHT * float(cleaned_this_step)

        # Less penalty when this step actually cleans (encourage coverage vs empty wandering).
        step_penalty = (
            self.STEP_PENALTY_ACTIVE if cleaned_this_step > 0 else self.STEP_PENALTY_IDLE
        )

        approach_reward = (
            self.APPROACH_DIRT_REWARD if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        )

        gained = max(0, self.battery - self._prev_battery)
        charge_reward = self.CHARGE_GAIN_COEF * min(gained, 80)

        br = float(self.battery) / float(max(self.battery_max, 1))
        approach_charger = 0.0
        if br < self.LOW_BATTERY_RATIO and self.nearest_charger_dist < self.last_nearest_charger_dist:
            approach_charger = self.APPROACH_CHARGER_REWARD

        low_battery_penalty = 0.0
        if br < self.CRITICAL_BATTERY_RATIO:
            low_battery_penalty = self.CRITICAL_BATTERY_STEP_PENALTY
        elif br < self.LOW_BATTERY_RATIO:
            low_battery_penalty = self.LOW_BATTERY_STEP_PENALTY

        env_r = self.ENV_REWARD_BLEND * self._frame_env_reward
        x, z = self.cur_pos
        revisit_penalty = 0.0
        if self.ENABLE_REVISIT_PENALTY and 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            repeat_times = max(0, int(self.visit_count[x, z]) - 1)
            revisit_penalty = self.REVISIT_PENALTY_COEF * min(repeat_times, self.REVISIT_PENALTY_CAP)

        return (
            cleaning_reward
            + step_penalty
            + approach_reward
            + charge_reward
            + approach_charger
            + low_battery_penalty
            + revisit_penalty
            + env_r
        )
