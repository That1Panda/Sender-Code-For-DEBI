import socket
import io
import struct
import time
import picamera

# Set up socket connection
server_address = ('192.168.0.143', 1234)  # Replace with your IP address and port number
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Set up PiCamera
with picamera.PiCamera(resolution='320x240', framerate=24) as camera:
    camera.resolution = (320, 240)  # Set the resolution of the camera
    camera.framerate = 24  # Set the frame rate of the camera
    time.sleep(2)  # Wait for the camera to warm up
    stream = io.BytesIO()
    
# Continuously capture and send images over UDP
    for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
        # Read the image data from the stream
        data = stream.getvalue()
        size = 1024
        img_bytes = [data[i:i+size] for i in range(0, len(data), size)]

        # Get the number of frames in the image
        n_frames = len(img_bytes)

        # Send the number of frames to the receiver
        client_socket.sendto(str(n_frames).encode(), server_address)

        # Send the image frames to the receiver
        for i in range(len(img_bytes)):
            frame = img_bytes[i]
            client_socket.sendto(frame, server_address)

        # # Pack the image data and send it over UDP
        # packer = struct.Struct('I {}s'.format(len(data)))
        # print(len(data))
        # packed_data = packer.pack(len(data), data)
        # client_socket.sendto(packed_data, server_address)
        # Reset the stream for the next image
        stream.seek(0)
        stream.truncate()