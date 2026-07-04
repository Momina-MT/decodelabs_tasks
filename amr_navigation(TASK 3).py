"""Project 3: Autonomous Mobile Robot (AMR) Navigation — ANIMATED (GUI window)
Run with:  python3 amr_navigation_gui.py
"""
import heapq, math, random, tkinter as tk
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
@dataclass(order=True)
class AStarNode:
    f: float
    g: float         = field(compare=False)
    row: int         = field(compare=False)
    col: int         = field(compare=False)
    parent: object   = field(compare=False, default=None)
class OccupancyGrid:
    def __init__(self, width_m=10.0, height_m=10.0, resolution=0.1):
        self.resolution = resolution
        self.cols = int(width_m / resolution)
        self.rows = int(height_m / resolution)
        self.grid = [[-1]*self.cols for _ in range(self.rows)]
    def world_to_grid(self, x, y):
        return int(y/self.resolution), int(x/self.resolution)
    def grid_to_world(self, r, c):
        return (c+0.5)*self.resolution, (r+0.5)*self.resolution
    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols
    def set(self, r, c, v):
        if self.in_bounds(r, c): self.grid[r][c] = v
    def get(self, r, c):
        return self.grid[r][c] if self.in_bounds(r, c) else -1
    def _bresenham(self, r0, c0, r1, c1):
        dr, dc = abs(r1-r0), abs(c1-c0)
        sr, sc = (1 if r1>r0 else -1), (1 if c1>c0 else -1)
        err = dr - dc
        r, c = r0, c0
        while True:
            yield r, c
            if r==r1 and c==c1: break
            e2 = 2*err
            if e2 > -dc: err -= dc; r += sr
            if e2 <  dr: err += dr; c += sc
    def integrate_lidar(self, pose, ranges, angle_min, angle_inc, range_max=5.0):
        r0, c0 = self.world_to_grid(pose.x, pose.y)
        for i, dist in enumerate(ranges):
            angle = pose.theta + angle_min + i*angle_inc
            effective = range_max if (dist >= range_max or math.isnan(dist)) else dist
            hx = pose.x + effective*math.cos(angle)
            hy = pose.y + effective*math.sin(angle)
            r1, c1 = self.world_to_grid(hx, hy)
            cells = list(self._bresenham(r0, c0, r1, c1))
            for r, c in cells[:-1]:
                if self.get(r, c) != 100: self.set(r, c, 0)
            if dist < range_max and not math.isnan(dist):
                self.set(r1, c1, 100)
            else:
                if self.get(r1, c1) != 100: self.set(r1, c1, 0)
    def inflate(self, cells=2):
        obs = [(r,c) for r in range(self.rows) for c in range(self.cols)
               if self.grid[r][c] == 100]
        for (r,c) in obs:
            for dr in range(-cells, cells+1):
                for dc in range(-cells, cells+1):
                    nr, nc = r+dr, c+dc
                    if self.in_bounds(nr, nc) and self.grid[nr][nc] == 0:
                        self.grid[nr][nc] = 50
    def passable(self, r, c): return self.get(r,c) != 100
    def cost(self, r, c):
        v = self.get(r,c)
        if v==100: return float('inf')
        if v==50:  return 5.0
        return 1.0


class AStarPlanner:
    """
    A* with Manhattan heuristic on a 4-connected occupancy grid.
    f(n) = g(n) + h(n); h = Manhattan distance [admissible on 4-grid]
    """
    MOVES = [(-1,0),(1,0),(0,-1),(0,1)]

    def __init__(self, grid): self.grid = grid; self.last_expanded = 0

    def _h(self, r, c, gr, gc): return abs(r-gr)+abs(c-gc)

    def plan(self, start_world, goal_world):
        sr, sc = self.grid.world_to_grid(*start_world)
        gr, gc = self.grid.world_to_grid(*goal_world)
        if not self.grid.in_bounds(sr,sc) or not self.grid.in_bounds(gr,gc):
            return []
        if not self.grid.passable(gr,gc):
            return []

        start = AStarNode(f=self._h(sr,sc,gr,gc), g=0.0, row=sr, col=sc)
        heap  = [start]
        best  = {(sr,sc): 0.0}
        expanded = 0

        while heap:
            cur = heapq.heappop(heap)
            expanded += 1
            if cur.g > best.get((cur.row,cur.col), float('inf')):
                continue
            if cur.row==gr and cur.col==gc:
                self.last_expanded = expanded
                return self._path(cur)
            for dr,dc in self.MOVES:
                nr,nc = cur.row+dr, cur.col+dc
                if not self.grid.in_bounds(nr,nc): continue
                if not self.grid.passable(nr,nc):  continue
                tg = cur.g + self.grid.cost(nr,nc)
                if tg < best.get((nr,nc), float('inf')):
                    best[(nr,nc)] = tg
                    h = self._h(nr,nc,gr,gc)
                    heapq.heappush(heap, AStarNode(f=tg+h, g=tg,
                                                   row=nr, col=nc, parent=cur))
        self.last_expanded = expanded
        return []

    def _path(self, node):
        pts = []
        while node:
            pts.append(self.grid.grid_to_world(node.row, node.col))
            node = node.parent
        pts.reverse()
        return pts


class AvoidanceController:
    """
    Reflex-level safety override.
      error = distance_to_obstacle - safe_distance
      speed = max_speed * tanh(error)   when error < threshold
    """
    def __init__(self, safe_dist=0.45, max_speed=0.3, threshold=1.0):
        self.safe_dist  = safe_dist
        self.max_speed  = max_speed
        self.threshold  = threshold

    def compute(self, ranges, range_max=5.0):
        n = len(ranges)
        arc = [r for r in ranges[n//2-20:n//2+20]
               if not math.isnan(r) and r < range_max]
        if not arc:
            return {"speed": self.max_speed, "blocked": False, "status": "CLEAR"}
        d = min(arc)
        error = d - self.safe_dist
        if error < self.threshold:
            speed = self.max_speed * math.tanh(max(0.0, error))
            blocked = speed < 0.01
            return {"speed": speed, "blocked": blocked,
                    "dist": d, "status": "STOPPED" if blocked else "DECELERATING"}
        return {"speed": self.max_speed, "blocked": False, "dist": d, "status": "CLEAR"}


class SimLiDAR:
    def __init__(self, beams=360, range_max=5.0, noise=0.015):
        self.beams     = beams
        self.range_max = range_max
        self.noise     = noise
        self.angle_min = -math.pi
        self.angle_inc = 2*math.pi/beams
        self.walls     = []

    def add_wall(self, x1,y1,x2,y2): self.walls.append(((x1,y1),(x2,y2)))

    def _intersect(self, ox,oy,dx,dy, x1,y1,x2,y2):
        ex,ey = x2-x1, y2-y1
        d = dx*ey - dy*ex
        if abs(d)<1e-10: return None
        t = ((x1-ox)*ey-(y1-oy)*ex)/d
        u = ((x1-ox)*dy-(y1-oy)*dx)/d
        return t if (t>=0 and 0<=u<=1) else None

    def scan(self, pose):
        out = []
        for i in range(self.beams):
            angle = pose.theta + self.angle_min + i*self.angle_inc
            dx,dy = math.cos(angle), math.sin(angle)
            best  = self.range_max
            for (x1,y1),(x2,y2) in self.walls:
                t = self._intersect(pose.x,pose.y,dx,dy,x1,y1,x2,y2)
                if t and t < best: best = t
            out.append(max(0.0, best + random.gauss(0,self.noise)))
        return out


def build_world(lidar):
    for w in [(0,0,10,0),(10,0,10,10),(10,10,0,10),(0,10,0,0)]:
        lidar.add_wall(*w)
    for w in [(2,0,2,6),(2,7,2,10),(4,2,4,10),(4,0,4,1),
              (6,0,6,4),(6,5,6,8),(8,2,8,10),(8,0,8,1),(3,3,5,3)]:
        lidar.add_wall(*w)
    return (0.5,0.5), (9.5,9.5)

CELL_PX   = 6     
BG        = "#05080a"
PANEL_BG  = "#0b1210"
LINE      = "#16231d"
TXT       = "#eafff3"
TXT_DIM   = "#6f8f85"
CYAN      = "#5fe1c9"
AMBER     = "#ffb454"
AMBER_DIM = "#5a4527"
GREEN     = "#7cff9e"
RED       = "#ff5f56"
FREE      = "#0e1e19"
UNKNOWN   = "#050706"

COLOR_OF = {-1: UNKNOWN, 0: FREE, 50: AMBER_DIM, 100: AMBER}


class AMRApp:
    def __init__(self, root):
        self.root = root
        root.title("AMR Navigation — Live")
        root.configure(bg=BG)
        root.resizable(True, True)

        self._fit_to_screen()
        self._build_ui()
        self.reset_sim()
        self.tick()

    # ── responsive sizing ─────────────────────────
    def _fit_to_screen(self):
        """Pick a cell pixel size (and panel width) so the whole window
        comfortably fits the user's screen, however small it is."""
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
end
        max_w = int(screen_w * 0.85)
        max_h = int(screen_h * 0.80)
        chrome_h = 150          
        panel_w  = 260          
        avail_for_grid_h = max_h - chrome_h
        avail_for_grid_w = max_w - panel_w - 40
        grid_dim = min(avail_for_grid_w, avail_for_grid_h)
        self.CELL_PX = max(2, min(6, grid_dim // 100))
        self.canvas_dim = self.CELL_PX * 100
        self.panel_w = panel_w
        self.log_height_lines = 12 if self.canvas_dim < 500 else 18
    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=12, pady=(10,4))
        tk.Label(header, text="AMR NAVIGATION — LIVE", fg=TXT, bg=BG,
                  font=("Consolas", 15, "bold")).pack(side="left")
        self.phase_lbl = tk.Label(header, text="Initializing", fg=CYAN, bg=BG,
                                    font=("Consolas", 11, "bold"))
        self.phase_lbl.pack(side="right")
        main = tk.Frame(self.root, bg=BG)
        main.pack(padx=12, pady=6)
        canvas_wrap = tk.Frame(main, bg=PANEL_BG, highlightbackground=LINE, highlightthickness=1)
        canvas_wrap.pack(side="left")
        self.canvas = tk.Canvas(canvas_wrap, width=self.canvas_dim, height=self.canvas_dim, bg=UNKNOWN, highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self.grid_img = tk.PhotoImage(width=1, height=1)   # placeholder, real one made in reset_sim
        self.grid_img_id = self.canvas.create_image(0, 0, anchor="nw")
        legend = tk.Frame(canvas_wrap, bg=PANEL_BG)
        legend.pack(fill="x", padx=6, pady=(0,6))
        for text, color in [("free", FREE), ("obstacle", AMBER), ("inflated", AMBER_DIM),
                             ("path", GREEN), ("dyn. obstacle", RED), ("robot", "#ffffff")]:
            item = tk.Frame(legend, bg=PANEL_BG)
            item.pack(side="left", padx=6)
            tk.Frame(item, bg=color, width=10, height=10).pack(side="left")
            tk.Label(item, text=" "+text, fg=TXT_DIM, bg=PANEL_BG,
                      font=("Consolas", 8)).pack(side="left")

        # side panel
        panel = tk.Frame(main, bg=PANEL_BG, highlightbackground=LINE,
                           highlightthickness=1, width=self.panel_w)
        panel.pack(side="left", fill="y", padx=(10,0))
        panel.pack_propagate(False)

        tk.Label(panel, text="TELEMETRY", fg=TXT_DIM, bg=PANEL_BG,
                  font=("Consolas", 9, "bold")).pack(anchor="w", padx=12, pady=(12,6))

        self.stat_labels = {}
        stat_grid = tk.Frame(panel, bg=PANEL_BG)
        stat_grid.pack(fill="x", padx=12)
        fields = ["Pose", "Heading", "Speed", "Nearest obstacle",
                  "Status", "Waypoint", "Full stops", "Re-routes"]
        for i, name in enumerate(fields):
            r, c = divmod(i, 2)
            cell = tk.Frame(stat_grid, bg=PANEL_BG)
            cell.grid(row=r, column=c, sticky="w", pady=4, padx=(0,10))
            tk.Label(cell, text=name.upper(), fg=TXT_DIM, bg=PANEL_BG,
                      font=("Consolas", 7)).pack(anchor="w")
            val = tk.Label(cell, text="—", fg=TXT, bg=PANEL_BG,
                             font=("Consolas", 12, "bold"))
            val.pack(anchor="w")
            self.stat_labels[name] = val

        tk.Label(panel, text="CONSOLE", fg=TXT_DIM, bg=PANEL_BG,
                  font=("Consolas", 9, "bold")).pack(anchor="w", padx=12, pady=(14,6))
        log_wrap = tk.Frame(panel, bg=PANEL_BG)
        log_wrap.pack(fill="both", expand=True, padx=12, pady=(0,12))
        self.log_text = tk.Text(log_wrap, bg="#080d0b", fg=TXT_DIM, insertbackground=TXT,
                                  font=("Consolas", 8), wrap="word", height=self.log_height_lines,
                                  relief="flat", state="disabled")
        self.log_text.pack(fill="both", expand=True)
        for tag, color in [("head", CYAN), ("good", GREEN), ("warn", AMBER), ("bad", RED)]:
            self.log_text.tag_configure(tag, foreground=color)

        # controls
        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=12, pady=(0,10))

        self.play_btn = tk.Button(controls, text="▶ Play", command=self.toggle_play,
                                    bg="#0e1815", fg=CYAN, activebackground=CYAN,
                                    activeforeground="#03110d", relief="flat",
                                    font=("Consolas", 10, "bold"), padx=14, pady=6,
                                    bd=0, highlightthickness=1, highlightbackground=LINE)
        self.play_btn.pack(side="left")

        restart_btn = tk.Button(controls, text="⟲ Restart", command=self.restart,
                                  bg="#0e1815", fg=TXT, activebackground=LINE,
                                  relief="flat", font=("Consolas", 10, "bold"),
                                  padx=14, pady=6, bd=0, highlightthickness=1,
                                  highlightbackground=LINE)
        restart_btn.pack(side="left", padx=8)

        tk.Label(controls, text="SIM SPEED", fg=TXT_DIM, bg=BG,
                  font=("Consolas", 9)).pack(side="left", padx=(18,6))
        self.speed_var = tk.IntVar(value=6)
        speed_scale = tk.Scale(controls, from_=1, to=25, orient="horizontal",
                                 variable=self.speed_var, bg=BG, fg=TXT_DIM,
                                 troughcolor=PANEL_BG, highlightthickness=0,
                                 length=140, showvalue=True, font=("Consolas", 8))
        speed_scale.pack(side="left")

        self.step_lbl = tk.Label(controls, text="0 / 0", fg=TXT_DIM, bg=BG,
                                   font=("Consolas", 9))
        self.step_lbl.pack(side="right")
        tk.Label(controls, text="STEP", fg=TXT_DIM, bg=BG,
                  font=("Consolas", 9)).pack(side="right", padx=(0,6))

    # ── logging ──────────────────────────────────
    def log(self, text, tag=None):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text+"\n", tag or ())
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ── simulation reset ─────────────────────────
    def reset_sim(self):
        random.seed(42)
        self.grid    = OccupancyGrid(10.0, 10.0, 0.1)
        self.lidar   = SimLiDAR(beams=360, range_max=5.0)
        self.start, self.goal = build_world(self.lidar)
        self.planner = AStarPlanner(self.grid)
        self.avoid   = AvoidanceController(safe_dist=0.45, max_speed=0.3, threshold=1.0)

        self.scan_poses = [
            Pose(0.5,0.5,0),      Pose(1,1,math.pi/4),
            Pose(3,1,0),          Pose(5,2,math.pi/2),
            Pose(5,5,math.pi),    Pose(3,7,-math.pi/4),
            Pose(7,7,0),          Pose(9,9,math.pi),
        ]

        self.phase       = "mapping"
        self.scan_idx    = 0
        self.path        = []
        self.pose        = Pose(*self.start)
        self.wp_idx      = 0
        self.step        = 0
        self.stops       = 0
        self.reroutes    = 0
        self.dyn_obs     = {150: 0.35, 300: 0.28}
        self.active_obs  = None
        self.obs_end_step= -1
        self.last_cmd    = {}
        self.done        = False
        self.reached     = False
        self.playing     = False
        self.play_btn.configure(text="▶ Play")

        self.grid_img = tk.PhotoImage(width=self.grid.cols, height=self.grid.rows)
        self.canvas.itemconfig(self.grid_img_id, image=self.grid_img)
        self._grid_dirty = True

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        self.log("="*44, "head")
        self.log("Project 3 — AMR Navigation | Tree Search (A*)", "head")
        self.log("DecodeLabs Industrial Training Kit | Batch 2026")
        self.log("="*44, "head")
        self.log("[SLAM] Mapping phase started...", "head")

        self.render()

    # ── one unit of simulation work ──────────────
    def advance_one(self):
        if self.done:
            return

        if self.phase == "mapping":
            p = self.scan_poses[self.scan_idx]
            rays = self.lidar.scan(p)
            self.grid.integrate_lidar(p, rays, self.lidar.angle_min,
                                        self.lidar.angle_inc, self.lidar.range_max)
            self.pose = p
            self._grid_dirty = True
            self.log(f"  Scan {self.scan_idx+1}/{len(self.scan_poses)}  "
                      f"({p.x:.1f},{p.y:.1f})  \u03b8={math.degrees(p.theta):.0f}\u00b0")
            self.scan_idx += 1
            if self.scan_idx >= len(self.scan_poses):
                self.phase = "inflating"
                self.log("[COSTMAP] Inflating obstacles (2-cell margin = 0.2 m)...", "warn")
            return

        if self.phase == "inflating":
            self.grid.inflate(2)
            self._grid_dirty = True
            self.phase = "planning"
            self.log(f"[A*] Planning {self.start} -> {self.goal} ...", "head")
            return

        if self.phase == "planning":
            path = self.planner.plan(self.start, self.goal)
            self.path = path
            if not path:
                self.log("[A*] FAILED — no path found!", "bad")
                self.phase = "done"; self.done = True
                return
            path_len = sum(math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
                           for i in range(1, len(path)))
            self.log(f"[A*] Path found — {self.planner.last_expanded} nodes expanded", "good")
            self.log(f"[A*] {len(path)} waypoints | {path_len:.2f} m", "good")
            self.log("[NAV] Executing...", "head")
            self.phase = "navigating"
            self.wp_idx = 0
            return

        if self.phase == "navigating":
            if self.wp_idx >= len(self.path):
                self._finish_nav()
                return

            step = self.step
            scan = self.lidar.scan(self.pose)

            if step in self.dyn_obs:
                self.active_obs = self.dyn_obs.pop(step)
                self.obs_end_step = step + 12
                self.log(f"   Step {step}: Dynamic obstacle at {self.active_obs:.2f} m!", "bad")
            if self.active_obs is not None and step >= self.obs_end_step:
                self.log(f"   Step {step}: Obstacle cleared — resuming...", "head")
                self.active_obs = None
            if self.active_obs is not None:
                mid = len(scan)//2
                for j in range(mid-15, mid+15):
                    if 0 <= j < len(scan):
                        scan[j] = min(scan[j], self.active_obs)

            cmd = self.avoid.compute(scan, self.lidar.range_max)
            self.last_cmd = cmd

            if cmd["blocked"]:
                self.stops += 1
                self.log(f"  STOPPED Step {step}: ({cmd['dist']:.2f} m) — re-planning...", "bad")
                self.grid.integrate_lidar(self.pose, scan, self.lidar.angle_min,
                                            self.lidar.angle_inc, self.lidar.range_max)
                self._grid_dirty = True
                new_path = self.planner.plan((self.pose.x, self.pose.y), self.goal)
                if new_path:
                    self.path, self.wp_idx = new_path, 0
                    self.reroutes += 1
                    self.log(f"          New path: {len(new_path)} waypoints", "good")
                else:
                    self.log("          Re-plan failed — holding.", "bad")
                self.step += 1
                return

            tx, ty = self.path[self.wp_idx]
            dx, dy = tx-self.pose.x, ty-self.pose.y
            dist = math.hypot(dx, dy)
            if dist < 0.12:
                self.wp_idx += 1
                self.step += 1
                return
            step_size = min(cmd["speed"]*0.1, dist)
            self.pose.x += (dx/dist)*step_size
            self.pose.y += (dy/dist)*step_size
            self.pose.theta = math.atan2(dy, dx)
            self.step += 1

            if self.step >= 2000:
                self._finish_nav()

    def _finish_nav(self):
        self.reached = math.hypot(self.pose.x-self.goal[0], self.pose.y-self.goal[1]) < 0.5
        self.phase = "done"
        self.done = True
        self.log("NAVIGATION COMPLETE", "head")
        self.log(f"Goal reached: {'YES' if self.reached else 'NO'}",
                  "good" if self.reached else "bad")
        self.log(f"Steps:{self.step}  Stops:{self.stops}  Re-routes:{self.reroutes}")
    def render(self):
        if self._grid_dirty:
            rows = []
            for r in range(self.grid.rows):
                row = self.grid.grid[r]
                rows.append("{" + " ".join(COLOR_OF[v] for v in row) + "}")
            self.grid_img.put(" ".join(rows))
            self._grid_dirty = False

        zoomed = self.grid_img.zoom(self.CELL_PX, self.CELL_PX)
        self.canvas.itemconfig(self.grid_img_id, image=zoomed)
        self._zoomed_ref = zoomed   # keep a reference so Tk doesn't garbage-collect it

        self.canvas.delete("overlay")
        self._draw_marker(self.start, CYAN, "S")
        self._draw_marker(self.goal, GREEN, "G")

        if self.path and len(self.path) > 1:
            pts = []
            for wx, wy in self.path:
                r, c = self.grid.world_to_grid(wx, wy)
                pts.extend([c*self.CELL_PX+self.CELL_PX/2, r*self.CELL_PX+self.CELL_PX/2])
            self.canvas.create_line(*pts, fill=GREEN, width=2, dash=(4,3), tags="overlay")

        if self.active_obs is not None and self.phase == "navigating":
            ox = self.pose.x + self.active_obs*math.cos(self.pose.theta)
            oy = self.pose.y + self.active_obs*math.sin(self.pose.theta)
            r, c = self.grid.world_to_grid(ox, oy)
            px, py = c*self.CELL_PX+self.CELL_PX/2, r*self.CELL_PX+self.CELL_PX/2
            self.canvas.create_oval(px-6, py-6, px+6, py+6, fill=RED, outline="",
                                      tags="overlay")

        self._draw_robot()
        self._update_stats()

    def _draw_marker(self, world_pt, color, label):
        r, c = self.grid.world_to_grid(*world_pt)
        px, py = c*self.CELL_PX+self.CELL_PX/2, r*self.CELL_PX+self.CELL_PX/2
        self.canvas.create_oval(px-4, py-4, px+4, py+4, fill=color, outline="",
                                  tags="overlay")
        self.canvas.create_text(px+10, py-8, text=label, fill=color,
                                  font=("Consolas", 9, "bold"), tags="overlay")

    def _draw_robot(self):
        r, c = self.grid.world_to_grid(self.pose.x, self.pose.y)
        px, py = c*self.CELL_PX+self.CELL_PX/2, r*self.CELL_PX+self.CELL_PX/2
        theta = self.pose.theta
        pts = []
        for ang_off, rad in [(0, 9), (2.5, 6), (-2.5, 6)]:
            a = theta + ang_off
            pts.extend([px+rad*math.cos(a), py+rad*math.sin(a)])
        self.canvas.create_polygon(*pts, fill="#ffffff", outline="", tags="overlay")

    def _update_stats(self):
        self.phase_lbl.configure(text={
            "mapping":"Mapping — SLAM scan", "inflating":"Building costmap",
            "planning":"A* planning", "navigating":"Navigating", "done":"Complete"
        }.get(self.phase, self.phase))

        cmd = self.last_cmd or {}
        self.stat_labels["Pose"].configure(text=f"{self.pose.x:.2f}, {self.pose.y:.2f}")
        self.stat_labels["Heading"].configure(text=f"{math.degrees(self.pose.theta):.0f}\u00b0")
        self.stat_labels["Speed"].configure(text=f"{cmd.get('speed',0):.3f} m/s")
        self.stat_labels["Nearest obstacle"].configure(
            text=f"{cmd['dist']:.2f} m" if "dist" in cmd else "—")

        status = cmd.get("status", "—")
        if self.done:
            status = "REACHED" if self.reached else "FAILED"
        color = {"STOPPED":RED, "FAILED":RED, "DECELERATING":AMBER,
                 "REACHED":GREEN, "CLEAR":TXT}.get(status, TXT)
        self.stat_labels["Status"].configure(text=status, fg=color)

        self.stat_labels["Waypoint"].configure(
            text=f"{min(self.wp_idx,len(self.path))}/{len(self.path)}" if self.path else "—")
        self.stat_labels["Full stops"].configure(text=str(self.stops),
                                                    fg=AMBER if self.stops else TXT)
        self.stat_labels["Re-routes"].configure(text=str(self.reroutes),
                                                   fg=AMBER if self.reroutes else TXT)

        if self.phase in ("navigating","done"):
            self.step_lbl.configure(text=f"{self.step} / ~2000")
        else:
            self.step_lbl.configure(text=f"{self.scan_idx} / {len(self.scan_poses)} scans")

    # ── controls ─────────────────────────────────
    def toggle_play(self):
        if self.done:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play")

    def restart(self):
        self.reset_sim()

    def tick(self):
        if self.playing and not self.done:
            for _ in range(self.speed_var.get()):
                if self.done:
                    break
                self.advance_one()
            if self.done:
                self.playing = False
                self.play_btn.configure(text="▶ Play")
        self.render()
        self.root.after(30, self.tick)


if __name__ == "__main__":
    root = tk.Tk()
    app = AMRApp(root)
    root.update_idletasks()
    # center the window on screen after it's sized itself
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"+{max(0,(sw-w)//2)}+{max(0,(sh-h)//2)}")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()