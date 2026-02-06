# parser.py
import struct
from config import (
    BLOCK_COUNT, BLOCK_SIZE,
    CHANNELS_PER_FIRING, CHANNEL_SIZE,
    MAX_VALID_DISTANCE
)

def azimuth_to_degrees(raw):
    return struct.unpack("<H", raw)[0] * 0.01

def distance_to_meters(raw):
    return struct.unpack("<H", raw)[0] * 0.0025

def is_valid_return(dist, intensity):
    return 0.05 < dist <= MAX_VALID_DISTANCE and intensity != 0

def parse_firing_channels(block):
    if block[0:2] != b'\xff\xee':
        raise ValueError("Bad block flag")

    channel_data = block[4:100]
    echo1, echo2 = [], []

    for ch in range(CHANNELS_PER_FIRING):
        offset = ch * CHANNEL_SIZE * 2
        dist1 = distance_to_meters(channel_data[offset:offset+2])
        inten1 = channel_data[offset+2]
        dist2 = distance_to_meters(channel_data[offset+3:offset+5])
        inten2 = channel_data[offset+5]
        echo1.append((dist1, inten1))
        echo2.append((dist2, inten2))

    return echo1, echo2

def interpolate_azimuths(az_current, az_next):
    delta = az_next - az_current
    if delta < 0:
        delta += 360.0
    return [(az_current + delta * ch / 32) % 360 for ch in range(32)]

def parse_payload(payload):
    firings = []
    blocks = [payload[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE] for i in range(BLOCK_COUNT)]
    block_azs = [azimuth_to_degrees(b[2:4]) for b in blocks]

    for i in range(0, BLOCK_COUNT - 2, 2):
        az_per_channel = interpolate_azimuths(block_azs[i], block_azs[i+2])

        e1_odd, e2_odd = parse_firing_channels(blocks[i])
        firings.append({
            "azimuth": az_per_channel[:16],
            "echo1": e1_odd,
            "echo2": e2_odd
        })

        e1_even, e2_even = parse_firing_channels(blocks[i+1])
        firings.append({
            "azimuth": az_per_channel[16:],
            "echo1": e1_even,
            "echo2": e2_even
        })

    return firings
