"""
Extract specific HDF5 files from the remote emg2pose tar via HTTP range requests.
Fetches large chunks to scan headers efficiently, then downloads only matching files.

Usage:
    PYTHONUNBUFFERED=1 /opt/anaconda3/envs/emg2pose/bin/python scripts/extract_train_user.py
"""

import os
import sys
import time

import requests

TAR_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
BLOCK_SIZE = 512
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB per HTTP request

WANTED = {
    # Training sessions (5 different stages, right hand)
    "emg2pose_data/2022-04-13-1649836800-04d0a-cv-emg-pose-demonstration@2-recording-1_right.hdf5",
    "emg2pose_data/2022-04-13-1649836800-04d0a-cv-emg-pose-demonstration@2-recording-10_right.hdf5",
    "emg2pose_data/2022-04-13-1649836800-04d0a-cv-emg-pose-demonstration@2-recording-11_right.hdf5",
    "emg2pose_data/2022-04-13-1649836800-04d0a-cv-emg-pose-demonstration@2-recording-12_right.hdf5",
    "emg2pose_data/2022-04-13-1649836800-04d0a-cv-emg-pose-demonstration@2-recording-14_right.hdf5",
    # Test sessions (2 stages, right hand)
    "emg2pose_data/2022-04-13-1649836800-04d0a-cv-emg-pose-demonstration@2-recording-16_right.hdf5",
    "emg2pose_data/2022-04-13-1649836800-62d94-cv-emg-pose-demonstration@2-recording-17_right.hdf5",
}

OUT_DIR = os.path.expanduser("~/emg2pose_train_user")


def fetch_range(start, end):
    r = requests.get(TAR_URL, headers={"Range": f"bytes={start}-{end}"})
    r.raise_for_status()
    return r.content


def format_size(nbytes):
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.1f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    return f"{nbytes / 1e3:.1f} KB"


def progress_bar(current, total, width=40, extra=""):
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(
        f"\r  [{bar}] {pct * 100:5.1f}% "
        f"({format_size(current)}/{format_size(total)}) {extra}  "
    )
    sys.stdout.flush()


def parse_header_at(buf, pos):
    """Parse a tar header from buffer at given position.
    Returns (name, file_size, typeflag, header_end_pos) or None if invalid."""
    if pos + BLOCK_SIZE > len(buf):
        return None
    header = buf[pos : pos + BLOCK_SIZE]
    if header == b"\0" * BLOCK_SIZE:
        return None

    name = header[0:100].split(b"\0")[0].decode("utf-8", errors="replace")
    prefix = header[345:500].split(b"\0")[0].decode("utf-8", errors="replace")
    if prefix:
        name = prefix + "/" + name

    size_field = header[124:136].split(b"\0")[0].decode().strip()
    file_size = int(size_field, 8) if size_field else 0
    typeflag = header[156:157]

    return name, file_size, typeflag


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    r = requests.head(TAR_URL)
    total_size = int(r.headers["Content-Length"])

    print(f"Tar size: {format_size(total_size)}")
    print(f"Looking for {len(WANTED)} files (training user 03d49b393e)")
    print(f"Output dir: {OUT_DIR}/")
    print(f"Chunk size: {format_size(CHUNK_SIZE)} per HTTP request")
    print()

    offset = 0
    found = 0
    files_scanned = 0
    start_time = time.time()

    while offset < total_size and found < len(WANTED):
        chunk_start = offset
        chunk_end = min(offset + CHUNK_SIZE - 1, total_size - 1)
        chunk = fetch_range(chunk_start, chunk_end)

        local_pos = 0
        while local_pos + BLOCK_SIZE <= len(chunk) and found < len(WANTED):
            result = parse_header_at(chunk, local_pos)
            if result is None:
                local_pos += BLOCK_SIZE
                offset = chunk_start + local_pos
                continue

            name, file_size, typeflag = result

            # Handle GNU long name extension
            if typeflag == b"L":
                name_blocks = (file_size + BLOCK_SIZE - 1) // BLOCK_SIZE
                name_data_start = local_pos + BLOCK_SIZE
                name_data_end = name_data_start + name_blocks * BLOCK_SIZE

                if name_data_end + BLOCK_SIZE <= len(chunk):
                    long_name = chunk[name_data_start:name_data_start + file_size]
                    name = long_name.split(b"\0")[0].decode("utf-8", errors="replace")
                    local_pos = name_data_end
                    result2 = parse_header_at(chunk, local_pos)
                    if result2 is None:
                        local_pos += BLOCK_SIZE
                        offset = chunk_start + local_pos
                        continue
                    _, file_size, _ = result2
                else:
                    # Long name spans chunk boundary — re-fetch from this offset
                    offset = chunk_start + local_pos
                    needed = name_data_end + BLOCK_SIZE
                    extra = fetch_range(chunk_start + local_pos, chunk_start + local_pos + needed - 1)
                    long_name = extra[BLOCK_SIZE : BLOCK_SIZE + file_size]
                    name = long_name.split(b"\0")[0].decode("utf-8", errors="replace")
                    real_header_pos = BLOCK_SIZE + name_blocks * BLOCK_SIZE
                    r2 = parse_header_at(extra, real_header_pos)
                    if r2 is None:
                        local_pos = name_data_end + BLOCK_SIZE
                        offset = chunk_start + local_pos
                        continue
                    _, file_size, _ = r2
                    local_pos = name_data_end

            files_scanned += 1
            file_data_blocks = (file_size + BLOCK_SIZE - 1) // BLOCK_SIZE
            file_total = BLOCK_SIZE + file_data_blocks * BLOCK_SIZE

            elapsed = time.time() - start_time
            rate = f"{files_scanned / elapsed:.0f} files/s" if elapsed > 0 else ""
            eta_s = (total_size - (chunk_start + local_pos)) / ((chunk_start + local_pos) / elapsed) if elapsed > 0 and (chunk_start + local_pos) > 0 else 0
            eta_min = eta_s / 60

            progress_bar(
                chunk_start + local_pos,
                total_size,
                extra=f"files={files_scanned} found={found}/{len(WANTED)} {rate} ETA={eta_min:.0f}m",
            )

            if name in WANTED:
                print()
                print(f"  >> FOUND: {os.path.basename(name)} ({format_size(file_size)})")

                data_abs_start = chunk_start + local_pos + BLOCK_SIZE
                data_abs_end = data_abs_start + file_size - 1

                # Check if file data is within current chunk
                data_local_start = local_pos + BLOCK_SIZE
                data_local_end = data_local_start + file_size
                if data_local_end <= len(chunk):
                    file_data = chunk[data_local_start:data_local_end]
                else:
                    file_data = fetch_range(data_abs_start, data_abs_end)

                out_path = os.path.join(OUT_DIR, os.path.basename(name))
                with open(out_path, "wb") as f:
                    f.write(file_data)
                print(f"     Saved: {out_path}")
                found += 1

            local_pos += file_total
            offset = chunk_start + local_pos

        if local_pos >= len(chunk):
            offset = chunk_start + local_pos

    elapsed = time.time() - start_time
    print()
    print()
    print(f"Done! Found {found}/{len(WANTED)} files in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Scanned {files_scanned} files in {format_size(offset)} of tar")
    print(f"Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
