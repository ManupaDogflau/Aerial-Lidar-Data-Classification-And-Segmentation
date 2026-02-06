# pcd_writer.py

def write_pcd(filename, points, colors=None):
    with open(filename, "w") as f:
        f.write("# .PCD v0.7\nVERSION 0.7\n")
        f.write("FIELDS x y z intensity\n" if colors is None else "FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\nHEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\nDATA ascii\n")

        for i, p in enumerate(points):
            if colors is None:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {p[3]:.1f}\n")
            else:
                r, g, b = colors[i]
                rgb = int(r*255)<<16 | int(g*255)<<8 | int(b*255)
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {rgb}\n")
