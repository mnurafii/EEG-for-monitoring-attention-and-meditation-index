import serial
import paho.mqtt.client as mqtt
import time
import json
import struct
import numpy as np

# ================= CONFIG =================
SERIAL_PORT = "/dev/rfcomm0"
BAUDRATE = 57600

MQTT_SERVER = "10.148.139.94"
MQTT_PORT   = 1882
MQTT_TOPIC  = "mindwave/eeg"

FS = 512               # sampling rate MindWave
FFT_WINDOW = 512       # 1 detik
EMA_ALPHA = 0.3        # smoothing band
PUBLISH_INTERVAL = 1.0 # detik
# =========================================


def clamp_0_100(v):
    return int(max(0, min(100, v)))


def fft_bandpower(raw):
    fft = np.fft.rfft(raw)
    freqs = np.fft.rfftfreq(len(raw), 1 / FS)
    psd = np.abs(fft) ** 2

    def band(low, high):
        idx = (freqs >= low) & (freqs <= high)
        return float(np.mean(psd[idx])) if np.any(idx) else 0.0

    return {
        "delta": band(0.5, 4),
        "theta": band(4, 8),
        "alpha": band(8, 13),
        "beta": band(13, 30),
        "gamma": band(30, 45),
    }


def relative_bandpower(bands):
    total = sum(bands.values()) + 1e-9
    return {k: (v / total) * 100 for k, v in bands.items()}


def compute_attention_meditation(rel):
    """
    Index berbasis EEG (0–100)
    """
    beta  = rel["beta"]
    alpha = rel["alpha"]
    theta = rel["theta"]

    eps = 1e-6

    # rasio dasar
    attention_raw  = beta / (alpha + theta + eps)
    meditation_raw = (alpha + theta) / (beta + eps)

    # scaling empiris (hasil trial MindWave)
    attention  = clamp_0_100((attention_raw / 2.5) * 100)
    meditation = clamp_0_100((meditation_raw / 3.0) * 100)

    return attention, meditation


# ===== MQTT =====
mqtt_client = mqtt.Client(protocol=mqtt.MQTTv311)
mqtt_client.connect(MQTT_SERVER, MQTT_PORT, 60)
mqtt_client.loop_start()
print("✅ MQTT connected")

# ===== SERIAL =====
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
print("✅ RFCOMM serial opened")
print("📡 Listening REAL EEG packets...")

buffer = bytearray()
raw_buffer = []
ema_bands = {}

last_publish = 0
last_poor_signal = None

try:
    while True:
        b = ser.read(1)
        if not b:
            continue

        buffer += b

        # sync AA AA
        if len(buffer) >= 2 and buffer[-2:] == b'\xAA\xAA':
            buffer = b'\xAA\xAA'

            plen_b = ser.read(1)
            if not plen_b:
                continue
            plen = plen_b[0]

            payload = ser.read(plen)
            if len(payload) != plen:
                continue

            ser.read(1)  # checksum ignore

            i = 0
            while i < plen:
                code = payload[i]
                i += 1

                if code == 0x02 and i < plen:
                    last_poor_signal = payload[i]
                    i += 1

                elif code == 0x80 and i < plen:
                    vlen = payload[i]
                    i += 1
                    if vlen == 2 and i + 2 <= plen:
                        raw = struct.unpack(">h", payload[i:i+2])[0]
                        raw_buffer.append(raw)
                        raw_buffer = raw_buffer[-FFT_WINDOW:]
                        i += 2
                    else:
                        i += vlen

                else:
                    if i < plen:
                        skip = payload[i]
                        i += 1 if skip < 0x80 else skip

        # ===== PROCESS & PUBLISH =====
        now = time.time()
        if len(raw_buffer) >= FFT_WINDOW and (now - last_publish) >= PUBLISH_INTERVAL:
            bands = fft_bandpower(np.array(raw_buffer))

            # EMA smoothing
            for k, v in bands.items():
                prev = ema_bands.get(k, v)
                ema_bands[k] = EMA_ALPHA * v + (1 - EMA_ALPHA) * prev

            rel = relative_bandpower(ema_bands)
            attention, meditation = compute_attention_meditation(rel)

            eeg_data = {
                "delta": round(rel["delta"], 2),
                "theta": round(rel["theta"], 2),
                "alpha": round(rel["alpha"], 2),
                "beta":  round(rel["beta"], 2),
                "gamma": round(rel["gamma"], 2),
                "attention": attention,
                "meditation": meditation,
                "poor_signal": last_poor_signal,
                "timestamp": now
            }

            mqtt_client.publish(MQTT_TOPIC, json.dumps(eeg_data))
            print("📤 EEG REAL:", eeg_data)

            last_publish = now

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

finally:
    ser.close()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("✅ Clean exit")
