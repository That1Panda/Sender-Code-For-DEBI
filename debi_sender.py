import socket
import cv2
import numpy as np
def main():
    # Define the IP address and port to send data to
    ip_address = "192.168.0.143"  # Replace with the IP address of the receiver
    port = 1234  # Replace with the port number used by the receiver

    # Create a socket to send data
    client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Open the laptop camera
    # open camera
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)

    # set dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        # cv2.imshow("IMG",frame)
        # if not ret:
        #     break

        # Encode the frame as a JPEG byte array
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, img_bytes = cv2.imencode(".jpg", frame, encode_param)

        # Apply tobytes to all img_bytes
        img_bytes = [i.tobytes() for i in img_bytes]

        # Concatenate all img_bytes
        img_bytes = b''.join(img_bytes)

        # Split img_bytes into 1000 bytes each
        size = 1024
        img_bytes = [img_bytes[i:i+size] for i in range(0, len(img_bytes), size)]

        # Get the number of frames in the image
        n_frames = len(img_bytes)

        # Send the number of frames to the receiver
        client_socket.sendto(str(n_frames).encode(), (ip_address, port))

        # Send the image frames to the receiver
        for i in range(len(img_bytes)):
            frame = img_bytes[i]
            print(len(frame))
            client_socket.sendto(frame, (ip_address, port))

        # Wait for a key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the socket
    cap.release()
    client_socket.close()


if __name__ == "__main__":
    main()
    