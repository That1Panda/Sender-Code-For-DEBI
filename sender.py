import socket

# Define the target address and port for the UDP connection
udp_target_host = "192.168.0.127"
udp_target_port = 1234

# Create a UDP socket and send a test message
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.sendto(b"Hello, world!", (udp_target_host, udp_target_port))

# Close the UDP socket
udp_socket.close()
