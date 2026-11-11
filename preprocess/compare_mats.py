"""Compare two PiP-format .mat outputs (reference vs fast reimplementation)."""
import sys
import h5py
import numpy as np


def load(path):
    f = h5py.File(path, "r")
    key = [k for k in f.keys() if k.lower().startswith("traj")][0]
    traj = np.transpose(f[key][:])
    ft = f["tracks"]
    cols, rows = ft.shape
    tracks = [[np.transpose(f[ft[j][i]][:]) for j in range(cols)] for i in range(rows)]
    return traj, tracks


def main(a_path, b_path):
    ta, tra = load(a_path)
    tb, trb = load(b_path)
    print(f"traj  A={ta.shape}  B={tb.shape}")
    ok = True
    if ta.shape != tb.shape:
        ok = False
        print("!! traj shape differs")
    else:
        # row order may differ, so compare as sorted sets of rows
        ka = ta[np.lexsort((ta[:, 2], ta[:, 1], ta[:, 0]))]
        kb = tb[np.lexsort((tb[:, 2], tb[:, 1], tb[:, 0]))]
        diff = np.abs(ka - kb)
        bad = np.argwhere(diff > 1e-3)
        print(f"traj max abs diff = {diff.max():.6g}, mismatching cells = {len(bad)}")
        if len(bad):
            ok = False
            for r, c in bad[:10]:
                print(f"   row {r} col {c}: A={ka[r, c]} B={kb[r, c]}")
    print(f"tracks A={len(tra)}x{len(tra[0])}  B={len(trb)}x{len(trb[0])}")
    ndiff = 0
    for i in range(min(len(tra), len(trb))):
        for j in range(min(len(tra[i]), len(trb[i]))):
            x, y = tra[i][j], trb[i][j]
            if x.shape != y.shape or (x.size and np.abs(x - y).max() > 1e-3):
                ndiff += 1
    print(f"tracks cells differing: {ndiff}")
    ok = ok and ndiff == 0
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
