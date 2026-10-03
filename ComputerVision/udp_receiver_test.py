import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.bind((UDP_IP, UDP_PORT))

print("✅ UDP Receiver started")
print(f"Listening on {UDP_IP}:{UDP_PORT}")
print("Waiting for data...\n")

while True:

    data, address = sock.recvfrom(1024)

    message = data.decode("utf-8")

    print("Received:", message)