
  PROJECT 2: Automated Quality Inspection 
CONFIG = {
    "FRAME_W"          : 900, "FRAME_H"          : 500,"BELT_Y1"          : 160,    # top of belt
    "BELT_Y2"          : 360,    # bottom of belt
    "INSPECTION_X"     : 450,    # camera inspection 
    "PART_SPEED"       : 3,      # pixels per frame
    "GAUSSIAN_KERNEL"  : (5, 5),
    "BINARY_THRESHOLD" : 50,     # gear=160 grey > bg=0 black → clean separation
    "AREA_THRESHOLD"   : 1260,   # perfect ~1548px, defective ~950px → split at 1260
    "GAP_DEPTH_MIN_PX" : 8,
    "EXPECTED_GAP_COUNT": 4,     # calibrated: perfect gear = 4 gaps, defective = 4 or 5
    "FPS_TARGET"       : 30,
    "SPAWN_INTERVAL_S" : 2.5,    # seconds between new parts
}
BELT_CX = (CONFIG["BELT_Y1"] + CONFIG["BELT_Y2"]) // 2   # vertical centre of belt
#  GEAR dynammics
def make_gear_mask(size, broken_tooth=None):
    """Return a binary mask of a gear (white on black)."""
    R      = size // 2
    canvas = np.zeros((size * 2, size * 2), np.uint8)
    CX = CY = size
    TEETH   = 24
    TOOTH_H = size // 6
    INNER_R = size // 3
    cv2.circle(canvas, (CX, CY), R, 255, -1)
    cv2.circle(canvas, (CX, CY), INNER_R, 0, -1)
    for i in range(TEETH):
        ang = np.deg2rad(360.0 / TEETH * i)
        tx  = int(CX + (R + TOOTH_H) * np.cos(ang))
        ty  = int(CY + (R + TOOTH_H) * np.sin(ang))
        hw  = int(np.pi * R / TEETH * 0.45)
 pts = np.array([[tx-hw,ty],[tx+hw,ty],  [tx+hw,ty-TOOTH_H],[tx-hw,ty-TOOTH_H]], np.int32)
   M   = cv2.getRotationMatrix2D((CX,CY), -360.0/TEETH*i, 1.0)
 pts = cv2.transform(pts.reshape(-1,1,2).astype(np.float32),  M).reshape(-1,2).astype(np.int32)
 cv2.fillPoly(canvas, [pts], 255)
    # Broken tooth carve a notch
    if broken_tooth is not None:
        ang = np.deg2rad(360.0 / TEETH * broken_tooth)
        nx  = int(CX + (R - size//12) * np.cos(ang))
        ny  = int(CY + (R - size//12) * np.sin(ang))
        cv2.circle(canvas, (nx, ny), size//8, 0, -1)
    # crop back to size×size
    return canvas[CY-R-TOOTH_H-2 : CY+R+TOOTH_H+2,
                  CX-R-TOOTH_H-2 : CX+R+TOOTH_H+2]
#  IPO PIPELINE (same as batch script)
def preprocess(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, CONFIG["GAUSSIAN_KERNEL"], 0) _, thresh = cv2.threshold(blurred, CONFIG["BINARY_THRESHOLD"],
                       255, cv2.THRESH_BINARY)
    return gray, blurred, thresh
def analyse(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0, 0, []
    c    = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    hull = cv2.convexHull(c, returnPoints=False)
    try:
        defects = cv2.convexityDefects(c, hull)
    except cv2.error:
        return c, area, 0, []

    deep = []
    if defects is not None:
        for row in defects:
            s,e,f,d = row[0]
            depth   = d / 256.0
            if depth > CONFIG["GAP_DEPTH_MIN_PX"]:
                deep.append((tuple(c[f][0]), depth))

    return c, area, len(deep), deep


def tolerance_gate(area, gap_count):
    return ("PASS" if area >= CONFIG["AREA_THRESHOLD"] and gap_count == CONFIG["EXPECTED_GAP_COUNT"]  else "FAIL")
#  PART CLASS
class ConveyorPart:
    PART_SIZE = 80   # rendered size on belt

    def __init__(self, part_id):
        self.id         = part_id
        self.x          = -self.PART_SIZE        # start off-screen left
        self.y          = BELT_CX
        self.defective  = random.random() < 0.5
        self.broken_idx = random.randint(0, 23) if self.defective else None
        self.inspected  = False
        self.verdict    = None
        self.finished   = False

        # pre-render the gear mask
        raw_mask = make_gear_mask(self.PART_SIZE // 2, self.broken_idx)
        h, w     = raw_mask.shape
        # create colour gear image (steel grey) + alpha
        self.gear_img  = np.zeros((h, w, 3), np.uint8)
        self.gear_img[raw_mask > 0] = (160, 165, 170)
        self.gear_mask = raw_mask

        # add slight noise
        noise = np.random.normal(0, 10, self.gear_img.shape).astype(np.int16)
        self.gear_img = np.clip(
            self.gear_img.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

        self.gh, self.gw = h, w

    def update(self):
        self.x += CONFIG["PART_SPEED"]
        if self.x > CONFIG["FRAME_W"] + self.PART_SIZE:
            self.finished = True

    def draw_on(self, frame):
        """Blit gear onto frame."""
        cx = self.x
        cy = self.y
        x1 = cx - self.gw // 2;  x2 = x1 + self.gw
        y1 = cy - self.gh // 2;  y2 = y1 + self.gh

        # clip to frame bounds
        fx1 = max(x1, 0);  fy1 = max(y1, 0)
        fx2 = min(x2, CONFIG["FRAME_W"]); fy2 = min(y2, CONFIG["FRAME_H"])
        if fx1 >= fx2 or fy1 >= fy2:
            return

        gx1 = fx1 - x1;  gy1 = fy1 - y1
        gx2 = gx1 + (fx2 - fx1); gy2 = gy1 + (fy2 - fy1)

        region = frame[fy1:fy2, fx1:fx2]
        gear   = self.gear_img[gy1:gy2, gx1:gx2]
        mask   = self.gear_mask[gy1:gy2, gx1:gx2]

        region[mask > 0] = gear[mask > 0]

        # draw verdict box after inspection
        if self.verdict is not None:
            color = (0,200,0) if self.verdict == "PASS" else (0,0,220)
            bx1 = max(cx - self.gw//2 - 4, 0)
            by1 = max(cy - self.gh//2 - 4, 0)
            bx2 = min(cx + self.gw//2 + 4, CONFIG["FRAME_W"])
            by2 = min(cy + self.gh//2 + 4, CONFIG["FRAME_H"])
            cv2.rectangle(frame, (bx1,by1),(bx2,by2), color, 2)
            cv2.putText(frame, self.verdict,
                        (bx1, by1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def inspect(self):
        """Run IPO pipeline on this part's gear image."""
        if self.inspected:
            return
        self.inspected = True

        # Inspect the pre-rendered gear image (grey gear on black background)
        # Threshold at 50 cleanly separates gear pixels (~160) from background (0)
        _, _, thresh = preprocess(self.gear_img)
        _, area, gap_count, _ = analyse(thresh)
        self.verdict = tolerance_gate(area, gap_count)


# ─────────────────────────────────────────────
#  CONVEYOR BELT RENDERER
# ─────────────────────────────────────────────

def draw_belt(frame):
    """Draw the conveyor belt background."""
    B1 = CONFIG["BELT_Y1"]; B2 = CONFIG["BELT_Y2"]
    W  = CONFIG["FRAME_W"]

    # belt body
    cv2.rectangle(frame, (0, B1), (W, B2), (50, 50, 50), -1)

    # belt stripes (moving)
    stripe_spacing = 60
    t = int(time.time() * CONFIG["PART_SPEED"] * 10) % stripe_spacing
    for x in range(-stripe_spacing + t, W + stripe_spacing, stripe_spacing):
        cv2.line(frame, (x, B1), (x + 30, B2), (70, 70, 70), 2)

    # belt edges
    cv2.rectangle(frame, (0, B1), (W, B1+8),  (30,30,30), -1)
    cv2.rectangle(frame, (0, B2-8),(W, B2),   (30,30,30), -1)

    # rollers
    for rx in [0, W]:
        cv2.ellipse(frame, (rx, (B1+B2)//2),
                    (20, (B2-B1)//2), 0, 0, 360, (80,80,80), -1)
        cv2.ellipse(frame, (rx, (B1+B2)//2),
                    (20, (B2-B1)//2), 0, 0, 360, (100,100,100), 3)


def draw_camera_rig(frame):
    """Draw the inspection camera above the belt."""
    IX = CONFIG["INSPECTION_X"]
    B1 = CONFIG["BELT_Y1"]

    # camera mount arm
    cv2.line(frame, (IX, 0), (IX, B1), (180,180,180), 3)
    # camera body
    cv2.rectangle(frame, (IX-25, 10), (IX+25, 55), (60,60,60), -1)
    cv2.rectangle(frame, (IX-25, 10), (IX+25, 55), (120,120,120), 2)
    # lens
    cv2.circle(frame, (IX, 45), 10, (30,30,30), -1)
    cv2.circle(frame, (IX, 45), 10, (0,200,255), 2)
    # laser scan lines
    cv2.line(frame, (IX, 55), (IX-40, B1), (0,200,255), 1)
    cv2.line(frame, (IX, 55), (IX+40, B1), (0,200,255), 1)
    # label
    cv2.putText(frame, "INSPECTION CAMERA",
                (IX-85, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0,200,255), 1)


def draw_hud(frame, total, passed, failed, fps):
    """Draw heads-up display panel."""
    H = CONFIG["FRAME_H"]; W = CONFIG["FRAME_W"]

    # top bar
    cv2.rectangle(frame, (0,0),(W, 0), (20,20,20), -1)

    # bottom panel
    cv2.rectangle(frame, (0, H-70),(W, H), (20,20,20), -1)

    texts = [
        (f"SYSTEM: ONLINE", (0,220,0)),
        (f"PARTS INSPECTED: {total}", (200,200,200)),
        (f"PASSED: {passed}", (0,200,0)),
        (f"FAILED: {failed}", (0,60,220)),
        (f"FPS: {fps:.0f}", (150,150,150)),
    ]
    x = 12
    for txt, col in texts:
        cv2.putText(frame, txt, (x, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        x += len(txt) * 9 + 20

    # accuracy
    acc = (passed/total*100) if total > 0 else 0
    cv2.putText(frame, f"ACCURACY: {acc:.0f}%",
                (W-170, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0,200,255), 2)

    # module label
    cv2.putText(frame, "PROJECT 2: AUTOMATED QUALITY INSPECTION | DecodeLabs 2026",
                (10, H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (120,120,120), 1)


# ─────────────────────────────────────────────
#  MAIN SIMULATION LOOP
# ─────────────────────────────────────────────

def run_simulation():
    print("\n" + "="*60)
    print("  CONVEYOR BELT INSPECTION SIMULATION")
    print("  Press Q to quit | Press SPACE to pause")
    print("="*60 + "\n")

    parts        = []
    part_counter = 0
    total = passed = failed = 0
    last_spawn   = time.time() - CONFIG["SPAWN_INTERVAL_S"]
    last_frame_t = time.time()
    paused       = False
    fps          = 30.0

    cv2.namedWindow("DecodeLabs — Conveyor Inspection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DecodeLabs — Conveyor Inspection", 900, 500)

    while True:
        now = time.time()
        dt  = now - last_frame_t
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 0.001))
        last_frame_t = now

        frame = np.full(
            (CONFIG["FRAME_H"], CONFIG["FRAME_W"], 3), 25, np.uint8)

        # spawn new part
        if not paused and (now - last_spawn) >= CONFIG["SPAWN_INTERVAL_S"]:
            part_counter += 1
            parts.append(ConveyorPart(part_counter))
            last_spawn = now

        # update & inspect parts
        if not paused:
            for p in parts:
                p.update()
                # inspect when crossing the camera line
                if not p.inspected and p.x >= CONFIG["INSPECTION_X"]:
                    p.inspect()
                    if p.verdict == "PASS": passed += 1
                    else:                   failed += 1
                    total += 1
                    icon = "✅" if p.verdict=="PASS" else "❌"
                    print(f"  {icon}  Part #{p.id:03d}  "
                          f"({'DEFECTIVE' if p.defective else 'PERFECT ':8})  "
                          f"→ {p.verdict}")

            parts = [p for p in parts if not p.finished]

        # render
        draw_belt(frame)
        draw_camera_rig(frame)

        for p in parts:
            p.draw_on(frame)

        # inspection zone vertical line
        IX = CONFIG["INSPECTION_X"]
        cv2.line(frame, (IX, CONFIG["BELT_Y1"]),
                         (IX, CONFIG["BELT_Y2"]),
                         (0, 200, 255), 1)

        draw_hud(frame, total, passed, failed, fps)

        if paused:
            cv2.putText(frame, "-- PAUSED --",
                        (CONFIG["FRAME_W"]//2 - 70, CONFIG["FRAME_H"]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 3)

        cv2.imshow("DecodeLabs — Conveyor Inspection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord(' '):
            paused = not paused

        # cap FPS
        elapsed = time.time() - now
        sleep   = max(0, 1.0/CONFIG["FPS_TARGET"] - elapsed)
        time.sleep(sleep)

    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"  Total Inspected : {total}")
    print(f"  Passed          : {passed}")
    print(f"  Failed          : {failed}")
    print(f"  Accuracy        : {(passed/(total or 1)*100):.1f}%  "
          f"(pass rate, not ground-truth)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    np.random.seed(None)   # random each run
    run_simulation()
