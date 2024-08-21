import scipy.io
import cv2
import numpy as np


def create_viewer_tool(gt, data, scale_factor=2):
    data = data.reshape((data.shape[0], data.shape[1], 1)) if len(data.shape) < 3 else data
    current_channel = 0
    total_channels = data.shape[2]

    def display_channel(channel_index):
        # Normalize the map and channel data for display
        normalized_map = cv2.normalize(gt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        channel_image = data[:, :, channel_index]
        normalized_channel = cv2.normalize(channel_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize images to make them larger
        resized_map = cv2.resize(normalized_map, (0, 0), fx=scale_factor, fy=scale_factor)
        resized_channel = cv2.resize(normalized_channel, (0, 0), fx=scale_factor, fy=scale_factor)

        # Concatenate map and channel images side by side
        combined_image = np.hstack((resized_map, resized_channel))

        # Display the combined image
        cv2.imshow(f'Channel', combined_image)
        cv2.resizeWindow(f'Channel', combined_image.shape[1], combined_image.shape[0])

    def navigate_channels(action):
        nonlocal current_channel
        if action == 'next':
            current_channel = (current_channel + 1) % total_channels
        elif action == 'previous':
            current_channel = (current_channel - 1) % total_channels
        display_channel(current_channel)

    # Initial display
    display_channel(current_channel)

    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('n'):  # Next channel
            navigate_channels('next')
        elif key == ord('p'):  # Previous channel
            navigate_channels('previous')

    cv2.destroyAllWindows()


# Usage Example
if __name__ == '__main__':
    # Load the .mat file
    img = scipy.io.loadmat('Data/Beach.mat')
    map_data = img['map']
    data = img['data']

    # Call the function to create the viewer tool with a larger display
    create_viewer_tool(map_data, data, scale_factor=3)
