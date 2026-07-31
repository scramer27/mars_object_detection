import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Find any .bin file inside fpga_payload
bin_files = glob.glob("fpga_payload/*.bin")

if not bin_files:
    print(
        "No .bin files found in fpga_payload/! Check your payload generator directory."
    )
else:
    bin_path = bin_files[0]
    print(f"Loading and checking: {bin_path}")

    # Read binary raw bytes
    raw_data = np.fromfile(bin_path, dtype=np.uint8)

    # Reshape from Planar (3, 256, 256) -> Interleaved (256, 256, 3) for display
    planar_img = raw_data.reshape((3, 256, 256))
    interleaved_img = np.transpose(planar_img, (1, 2, 0))

    # Render image
    plt.imshow(interleaved_img)
    plt.title(f"Reconstructed: {os.path.basename(bin_path)}")
    plt.axis("off")
    plt.show()