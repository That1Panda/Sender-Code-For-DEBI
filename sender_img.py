import socket

# Define the target address and port for the UDP connection
udp_target_host = "192.168.0.127"
udp_target_port = 1234

# Define the path of the image to send
image_path = "../Projects/IMG_4895.jpg"

# Open the image file and read its contents
with open(image_path, "rb") as file:
    image_data = file.read()

# Create a UDP socket and send the image data
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.sendto(image_data, (udp_target_host, udp_target_port))

# Close the UDP socket
udp_socket.close()
