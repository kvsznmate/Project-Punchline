"""
probe7.py — deep-probe specific input devices to find exactly what channel
counts / samplerates they will open at. Prints each device's own reported
capabilities plus concrete open attempts at 16000 Hz and its native rate.

Run (from src/, livesounds env active):
    python probe7.py                     # probes devices 7, 1, 8, 0
    python probe7.py --devices 7         # just one
"""
import argparse
import sounddevice as sd


def try_open(device, sr, ch):
    try:
        s = sd.InputStream(samplerate=sr, blocksize=1600, channels=ch,
                           device=device, dtype="float32")
        s.start(); s.stop(); s.close()
        return "OK"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, nargs="+", default=[7, 1, 8, 0])
    args = ap.parse_args()

    hostapis = sd.query_hostapis()
    for d in args.devices:
        try:
            info = sd.query_devices(d)
        except Exception as e:
            print(f"\n=== device {d}: cannot query ({e}) ===")
            continue
        api = hostapis[info["hostapi"]]["name"]
        print(f"\n=== device {d}: {info['name']} ===")
        print(f"    host API           : {api}")
        print(f"    max_input_channels : {info['max_input_channels']}")
        print(f"    default_samplerate : {info['default_samplerate']}")
        for sr in (16000, int(info['default_samplerate'])):
            for ch in (1, 2):
                if ch > info["max_input_channels"]:
                    continue
                print(f"    open sr={sr:>5} ch={ch}: {try_open(d, sr, ch)}")


if __name__ == "__main__":
    main()
