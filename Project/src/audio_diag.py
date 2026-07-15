"""
audio_diag.py — diagnose why an input device won't open.

Lists every input-capable device with its host API and channel count, then
tries to actually OPEN each candidate at the settings the pipeline uses
(samplerate + channels), reporting which succeed and the exact error for those
that fail. This turns a cryptic PortAudio "-9996 Invalid device" into a concrete
answer: wrong index, wrong channel count, or wrong samplerate.

Run (from src/, with the livesounds env active):
    python audio_diag.py                 # probe at 16000 Hz, 1 channel
    python audio_diag.py --samplerate 44100 --channels 2
"""
import argparse
import sounddevice as sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samplerate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    args = ap.parse_args()

    hostapis = sd.query_hostapis()
    devices = sd.query_devices()

    print(f"Probing input devices at samplerate={args.samplerate}, "
          f"channels={args.channels}\n")
    print(f"{'idx':>3}  {'in':>2}  {'API':<16}  {'default_sr':>10}  name")
    print("-" * 80)

    openable = []
    for idx, dev in enumerate(devices):
        nin = dev.get("max_input_channels", 0)
        if nin <= 0:
            continue
        api = hostapis[dev["hostapi"]]["name"]
        sr = dev.get("default_samplerate")
        name = dev.get("name", "")
        print(f"{idx:>3}  {nin:>2}  {api:<16}  {str(sr):>10}  {name}")

        # Try opening at requested channels; if that fails and the device has
        # more channels, retry at the device's native channel count.
        tried = []
        for ch in {args.channels, nin}:
            if ch < 1 or ch > nin:
                continue
            try:
                s = sd.InputStream(samplerate=args.samplerate, blocksize=1600,
                                   channels=ch, device=idx, dtype="float32")
                s.start(); s.stop(); s.close()
                print(f"      -> OPENS OK at channels={ch}")
                openable.append((idx, name, api, ch))
                break
            except Exception as e:
                tried.append(f"channels={ch}: {type(e).__name__} {e}")
        else:
            for t in tried:
                print(f"      -> FAIL {t}")

    print("\n=== SUMMARY: devices that opened ===")
    if not openable:
        print("NONE opened at these settings. Try --samplerate 44100 or "
              "--channels 2.")
    for idx, name, api, ch in openable:
        print(f"  index {idx}  ({api}, {ch}ch)  {name}")
    print("\nStereo Mix / loopback candidates that opened are your best pick.")


if __name__ == "__main__":
    main()
