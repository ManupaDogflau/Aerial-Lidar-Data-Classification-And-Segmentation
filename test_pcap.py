from scapy.all import PcapReader

count = 0

with PcapReader("lidar_packets.pcap") as p:
    for pkt in p:
        count += 1
        if count % 1000 == 0:
            print(count)

print("Total:", count)
