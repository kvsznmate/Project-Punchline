"""
levelmeter.py — show the input level of a capture device, to confirm it is
actually receiving audio (not silence). Opens the device the same way the
pipeline does and prints RMS + peak per chunk.

Run (from src/, livesounds env, with audio PLAYING):
    python levelmeter.py --device 1
    python levelmeter.py --device 1 --samplerate 16000 --channels 1
"""
import argparse
import time
import numpy as np
import sounddevice as sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    print(f"Listening on device {args.device} at {args.samplerate} Hz, "
          f"{args.channels}ch for {args.seconds}s.")
    print("PLAY SOME AUDIO NOW. Watch the level — bars should move if audio "
          "is being captured.\n")

    peak_seen = 0.0
    heard = False

    def cb(indata, frames, tinfo, status):
        nonlocal peak_seen, heard
        x = indata.reshape(-1) if indata.ndim > 1 else indata
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        peak_seen = max(peak_seen, peak)
        if rms > 0.005:
            heard = True
        bar = "#" * min(50, int(rms * 500))
        print(f"rms={rms:0.4f} peak={peak:0.4f} |{bar:<50}|")

    with sd.InputStream(samplerate=args.samplerate, blocksize=args.samplerate // 4,
                        channels=args.channels, device=args.device,
                        dtype="float32", callback=cb):
        time.sleep(args.seconds)

    print(f"\nMax peak seen: {peak_seen:0.4f}")
    if heard:
        print("RESULT: audio WAS captured. The device is working.")
    else:
        print("RESULT: only silence/near-silence. The device is NOT receiving "
              "your playback audio — check Windows routing / default device.")


if __name__ == "__main__":
    main()
