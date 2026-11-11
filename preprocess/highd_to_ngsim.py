"""Convert the raw highD recordings into the per-direction NGSIM-format CSVs.

highD ships 60 recordings of a straight motorway filmed from above, with both
carriageways in one file: `x` runs along the road, `y` across it, and vehicles in
the upper lanes travel towards -x while the lower lanes travel towards +x. The
NGSIM pipeline instead expects one file per travel direction, with `Local Y`
increasing along travel, distances in feet, and a 10 Hz frame clock.

Three things therefore have to happen here, and each is a place where highD and
NGSIM genuinely disagree rather than a matter of taste:

  * Direction split and mirroring. Each recording becomes `NN-fwd.csv` (lower
    lanes) and `NN-bck.csv` (upper lanes); the backward side is mirrored in both
    axes so that travel is +Y and "left of the driver" is -X on both sides, which
    is what the lane numbering and the social grid assume.
  * Resampling 25 Hz -> 10 Hz. NGSIM is 10 Hz and the whole downstream pipeline
    (3 s history = 30 frames, 5 s future = 50 frames, maneuver windows of +-40)
    is written in those units. 10 Hz sample k sits at highD frame 1 + 2.5k, i.e.
    exactly on a frame for even k and exactly halfway between two for odd k, so
    the resampling is an exact linear interpolation rather than a fit.
  * Units and reference point. highD is metric and reports the bounding box
    corner; NGSIM is in feet and the MATLAB stage subtracts half a vehicle length
    from Local Y to recover the centre. Local Y is therefore written as the front
    of the vehicle, so that subtraction lands on the centre.

Output columns are the 14 of the classic per-period NGSIM txt, in that order.
Columns the pipeline never reads (global time and coordinates) are written as 0.
"""
import argparse
import os

import numpy as np
import pandas as pd

M_TO_FT = 3.280839895
CLASS_LABEL = {"Car": 2, "Truck": 3}          # NGSIM: 1 motorcycle, 2 auto, 3 truck
COLS = ["vehicle_id", "frame_id", "total_frames", "global_time", "local_x", "local_y",
        "global_x", "global_y", "v_length", "v_width", "v_class", "v_vel", "v_acc",
        "lane_id"]


def resample_track(df, fps_in, fps_out):
    """Linear interpolation onto the output frame clock, shared by all vehicles.

    The clock has to be global to the recording: neighbour lookup later matches
    vehicles by frame id, so two tracks must land on the same instants.
    """
    step = fps_in / fps_out
    f0, f1 = df["frame"].iloc[0], df["frame"].iloc[-1]
    k0 = int(np.ceil((f0 - 1) / step))
    k1 = int(np.floor((f1 - 1) / step))
    if k1 < k0:
        return None
    k = np.arange(k0, k1 + 1)
    at = 1.0 + k * step
    src = df["frame"].to_numpy(dtype=np.float64)
    out = {"frame_id": k + 1}
    for c in ("x", "y", "xVelocity", "yVelocity", "xAcceleration", "laneId"):
        out[c] = np.interp(at, src, df[c].to_numpy(dtype=np.float64))
    out["laneId"] = np.rint(out["laneId"])      # a lane id is not interpolable
    return pd.DataFrame(out)


def convert_recording(rec_dir, rec_id, fps_out, out_dir):
    tracks = pd.read_csv(os.path.join(rec_dir, "%02d_tracks.csv" % rec_id))
    meta = pd.read_csv(os.path.join(rec_dir, "%02d_tracksMeta.csv" % rec_id))
    rec = pd.read_csv(os.path.join(rec_dir, "%02d_recordingMeta.csv" % rec_id))
    fps_in = float(rec["frameRate"].iloc[0])

    info = meta.set_index("id")[["class", "drivingDirection", "width", "height"]]
    written = []
    for direction, tag in ((2, "fwd"), (1, "bck")):
        ids = info.index[info["drivingDirection"] == direction]
        sub = tracks[tracks["id"].isin(ids)]
        if sub.empty:
            continue
        rows = []
        for vid, g in sub.groupby("id", sort=True):
            r = resample_track(g.sort_values("frame"), fps_in, fps_out)
            if r is None:
                continue
            length, height = info.at[vid, "width"], info.at[vid, "height"]
            # highD reports the upper-left corner of the box in image axes.
            cx = r["x"] + 0.5 * length
            cy = r["y"] + 0.5 * height
            if direction == 2:                      # travels +x
                long_c, lat, vel, acc = cx, cy, r["xVelocity"], r["xAcceleration"]
            else:                                   # travels -x, mirror both axes
                long_c, lat, vel, acc = -cx, -cy, -r["xVelocity"], -r["xAcceleration"]
            r = r.assign(vehicle_id=vid,
                         # front of the vehicle: the MATLAB stage subtracts half a
                         # length from this column to get back to the centre
                         local_y=(long_c + 0.5 * length) * M_TO_FT,
                         local_x=lat * M_TO_FT,
                         v_length=length * M_TO_FT,
                         v_width=height * M_TO_FT,
                         v_class=CLASS_LABEL.get(info.at[vid, "class"], 2),
                         v_vel=vel * M_TO_FT,
                         v_acc=acc * M_TO_FT)
            rows.append(r)
        if not rows:
            continue
        d = pd.concat(rows, ignore_index=True)

        # Lane 1 is the leftmost lane as the driver sees it. After mirroring, left
        # is -X on both carriageways, so ordering the raw ids by mean X gives it.
        order = d.groupby("laneId")["local_x"].mean().sort_values().index
        d["lane_id"] = d["laneId"].map({old: i + 1 for i, old in enumerate(order)})
        # Absolute offsets are arbitrary; the grid and the model only use
        # differences. Keep both axes positive, as the NGSIM files are.
        d["local_x"] -= d["local_x"].min()
        d["local_y"] -= d["local_y"].min()
        d["total_frames"] = d.groupby("vehicle_id")["frame_id"].transform("size")
        for c in ("global_time", "global_x", "global_y"):
            d[c] = 0.0

        d = d.sort_values(["vehicle_id", "frame_id"])[COLS]
        path = os.path.join(out_dir, "%02d-%s.csv" % (rec_id, tag))
        np.savetxt(path, d.to_numpy(), fmt="%.4f", delimiter=",")
        written.append((path, len(d), d["vehicle_id"].nunique(), len(order)))
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--src", default=os.path.join(here, "../data/dataset/highd/data"))
    ap.add_argument("--dst", default=os.path.join(here, "../data/raw_highd"))
    ap.add_argument("--fps_out", type=float, default=10.0)
    ap.add_argument("--recordings", type=int, nargs="*", default=None)
    a = ap.parse_args()

    os.makedirs(a.dst, exist_ok=True)
    ids = a.recordings or sorted(
        int(f[:2]) for f in os.listdir(a.src) if f.endswith("_tracks.csv"))
    total = 0
    for rec_id in ids:
        for path, n, veh, lanes in convert_recording(a.src, rec_id, a.fps_out, a.dst):
            print("  %s: %d rows, %d vehicles, %d lanes" % (os.path.basename(path), n, veh, lanes))
            total += n
    print("done: %d recordings -> %d rows at %g Hz" % (len(ids), total, a.fps_out))


if __name__ == "__main__":
    main()
