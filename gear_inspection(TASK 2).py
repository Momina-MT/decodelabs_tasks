"""
  PROJECT 2: Automated Quality Inspection (Computer Vision)
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
CONFIG = {
    "GAUSSIAN_KERNEL"  : (5, 5),   # must be odd × odd
    "BINARY_THRESHOLD" : 127,       # 0-255; lower captures darker parts
    "AREA_THRESHOLD"   : 64950,     # min pixel area for a perfect gear
    "GAP_DEPTH_MIN_PX" : 20,        # convexity gap depth threshold (px)
    "EXPECTED_GAP_COUNT": 7,        # expected # of large gaps on perfect gear
    "OUTPUT_DIR"       : "inspection_results",
}
def _draw_gear(canvas, broken_tooth=None):
    """
    Draw a gear onto canvas (in-place).
    broken_tooth: int index (0..TEETH-1) of the missing tooth, or None.
    """
    SIZE = canvas.shape[0]
    CX = CY = SIZE // 2
    OUTER_R = 140; INNER_R = 55; TEETH = 24; TOOTH_H = 22

    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(mask, (CX, CY), OUTER_R, 255, -1)   # gear body
    cv2.circle(mask, (CX, CY), INNER_R, 0,   -1)   # centre hole

    for i in range(TEETH):
        ang  = np.deg2rad(360.0 / TEETH * i)
        tx   = int(CX + (OUTER_R + TOOTH_H) * np.cos(ang))
        ty   = int(CY + (OUTER_R + TOOTH_H) * np.sin(ang))
        hw   = int(np.pi * OUTER_R / TEETH * 0.45)

        pts  = np.array([[tx-hw, ty], [tx+hw, ty],
                         [tx+hw, ty-TOOTH_H], [tx-hw, ty-TOOTH_H]], np.int32)
        M    = cv2.getRotationMatrix2D((CX, CY), -360.0 / TEETH * i, 1.0)
        pts  = cv2.transform(
            pts.reshape(-1, 1, 2).astype(np.float32), M
        ).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Defect: carve a notch into the rim at the broken-tooth position
    # (guarantees measurable area loss at any angular position)
    if broken_tooth is not None:
        ang = np.deg2rad(360.0 / TEETH * broken_tooth)
        nx  = int(CX + (OUTER_R - 12) * np.cos(ang))
        ny  = int(CY + (OUTER_R - 12) * np.sin(ang))
        cv2.circle(mask, (nx, ny), 18, 0, -1)

    canvas[mask == 255] = (180, 180, 180)


def generate_dataset(output_dir):
    """Generate 20 synthetic gear images and save to output_dir/dataset/."""
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    for i in range(1, 21):
        canvas = np.full((400, 400, 3), 40, dtype=np.uint8)

        if i <= 10:
            _draw_gear(canvas)
            filename = f"part_{i:02d}_perfect.jpg"
        else:
            broken   = np.random.randint(0, 24)
            _draw_gear(canvas, broken_tooth=broken)
            filename = f"part_{i:02d}_defective.jpg"

        # add realistic sensor noise
        noise  = np.random.normal(0, 8, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dataset_dir, filename), canvas)

    print(f"[DATASET] 20 synthetic gear images → {dataset_dir}")
    return dataset_dir
def preprocess(img):
   
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, CONFIG["GAUSSIAN_KERNEL"], 0)
    _, thresh = cv2.threshold(
        blurred, CONFIG["BINARY_THRESHOLD"], 255, cv2.THRESH_BINARY)
    return gray, blurred, thresh
def analyse_contours(thresh):
    
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0, 0, []

    gear_contour = max(contours, key=cv2.contourArea)
    area         = cv2.contourArea(gear_contour)

    # Convex hull indices (not points) — required for convexityDefects
    hull_idx = cv2.convexHull(gear_contour, returnPoints=False)

    try:
        defects_raw = cv2.convexityDefects(gear_contour, hull_idx)
    except cv2.error:
        return gear_contour, area, 0, []

    deep_defects = []
    if defects_raw is not None:
        for row in defects_raw:
            s, e, f, d_raw = row[0]
            depth_px = d_raw / 256.0            # ← critical fix
            if depth_px > CONFIG["GAP_DEPTH_MIN_PX"]:
                far_pt = tuple(gear_contour[f][0])
                deep_defects.append((far_pt, depth_px))

    return gear_contour, area, len(deep_defects), deep_defects


# ─────────────────────────────────────────────────────────────
#  PHASE 3 — OUTPUT: Tolerance Gate (Decide & Act)
# ─────────────────────────────────────────────────────────────

def tolerance_gate(area, gap_count):
    
    area_ok = area  >= CONFIG["AREA_THRESHOLD"]
    gaps_ok = gap_count == CONFIG["EXPECTED_GAP_COUNT"]
    return "PASS" if (area_ok and gaps_ok) else "FAIL"


def render_verdict(img, gear_contour, area, gap_count,
                   deep_defects, verdict, part_id):
    out = img.copy()

    if gear_contour is None:
        cv2.putText(out, "ERROR: No part detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out

    # gear boundary in blue
    cv2.drawContours(out, [gear_contour], -1, (255, 100, 0), 2)

    if verdict == "FAIL":
        # red bounding boxes around each detected defect point
        for far_pt, depth in deep_defects:
            bx, by = far_pt
            half   = int(depth * 1.2)
            tl = (max(bx - half, 0),          max(by - half, 0))
            br = (min(bx + half, img.shape[1]), min(by + half, img.shape[0]))
            cv2.rectangle(out, tl, br, (0, 0, 255), 2)
            cv2.circle(out, far_pt, 5, (0, 255, 255), -1)
            cv2.putText(out, f"{depth:.1f}px",
                        (bx + 6, by - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 0, 180), -1)
        cv2.putText(out,
            f"[!] FAIL  |  {part_id}  |  "
            f"Area={int(area)}  Gaps={gap_count}",
            (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    else:
        cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 140, 0), -1)
        cv2.putText(out,
            f"[OK] PASS  |  {part_id}  |  "
            f"Area={int(area)}  Gaps={gap_count}",
            (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    cv2.putText(out, f"ID: {part_id}",
                (8, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    return out


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE — single image
# ─────────────────────────────────────────────────────────────

def inspect_image(img_path, output_dir):
    """Run the full IPO pipeline on one image. Returns result dict."""
    part_id = os.path.splitext(os.path.basename(img_path))[0]
    img     = cv2.imread(img_path)
    if img is None:
        return {"part_id": part_id, "verdict": "ERROR",
                "area_px": 0, "gap_count": 0, "defect_points": 0}

    # ── PHASE 1 ──
    gray, blurred, thresh = preprocess(img)

    # ── PHASE 2 ──
    gear_contour, area, gap_count, deep_defects = analyse_contours(thresh)

    # ── PHASE 3 ──
    verdict   = tolerance_gate(area, gap_count)
    annotated = render_verdict(img, gear_contour, area, gap_count,
                               deep_defects, verdict, part_id)

    # save annotated result
    ann_dir = os.path.join(output_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)
    cv2.imwrite(os.path.join(ann_dir, f"result_{part_id}.jpg"), annotated)

    # save pipeline stages (for report / debugging)
    stg = os.path.join(output_dir, "pipeline_stages", part_id)
    os.makedirs(stg, exist_ok=True)
    cv2.imwrite(os.path.join(stg, "1_gray.jpg"),    gray)
    cv2.imwrite(os.path.join(stg, "2_blurred.jpg"), blurred)
    cv2.imwrite(os.path.join(stg, "3_thresh.jpg"),  thresh)
    cv2.imwrite(os.path.join(stg, "4_result.jpg"),  annotated)

    return {
        "part_id"      : part_id,
        "verdict"      : verdict,
        "area_px"      : int(area),
        "gap_count"    : gap_count,
        "defect_points": len(deep_defects),
    }


def run_batch(dataset_dir, output_dir):
    """Inspect all images; print live results; write JSON report."""
    image_files = sorted([
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    results  = []
    passed   = failed = 0
    start_ts = datetime.now()

    print("\n" + "═" * 64)
    print("  DECODELABS — GEAR INSPECTION SYSTEM v1.0")
    print(f"  Module: GEAR_INSPECTION_V1.0  |  Batch: {len(image_files)} parts")
    print("═" * 64)

    for img_path in image_files:
        r    = inspect_image(img_path, output_dir)
        results.append(r)
        icon = "✅" if r["verdict"] == "PASS" else "❌"
        print(f"  {icon}  {r['part_id']:<36} → {r['verdict']}"
              f"  (area={r['area_px']}  gaps={r['gap_count']})")
        if r["verdict"] == "PASS": passed += 1
        else:                      failed += 1

    elapsed = (datetime.now() - start_ts).total_seconds()

    # accuracy vs ground-truth embedded in filename
    correct = sum(
        1 for r in results
        if ("perfect"   in r["part_id"] and r["verdict"] == "PASS") or
           ("defective" in r["part_id"] and r["verdict"] == "FAIL")
    )
    accuracy = (correct / len(results)) * 100 if results else 0

    summary = {
        "system"            : "GEAR_INSPECTION_V1.0",
        "timestamp"         : start_ts.isoformat(),
        "total_parts"       : len(results),
        "passed"            : passed,
        "failed"            : failed,
        "correct"           : correct,
        "accuracy_pct"      : round(accuracy, 1),
        "processing_time_s" : round(elapsed, 3),
        "config"            : {k: str(v) for k, v in CONFIG.items()},
        "results"           : results,
    }

    report_path = os.path.join(output_dir, "inspection_report.json")
    with open(report_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "═" * 64)
    print(f"  SYSTEM STATUS : ONLINE")
    print(f"  PASSED        : {passed} / {len(results)}")
    print(f"  FAILED        : {failed} / {len(results)}")
    print(f"  ACCURACY      : {accuracy:.1f}%")
    print(f"  PROCESSING    : {elapsed:.3f}s")
    print(f"  REPORT        → {report_path}")
    print("═" * 64 + "\n")
    return summary


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    OUT = CONFIG["OUTPUT_DIR"]
    os.makedirs(OUT, exist_ok=True)

    # 1. Generate validation dataset
    dataset_dir = generate_dataset(OUT)

    # 2. Run full inspection pipeline
    run_batch(dataset_dir, OUT)

    print(f"Annotated results  → {OUT}/annotated/")
    print(f"Pipeline stages    → {OUT}/pipeline_stages/")
    print(f"JSON report        → {OUT}/inspection_report.json")
