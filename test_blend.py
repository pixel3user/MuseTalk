import numpy as np
from musetalk.utils.blending import get_image_blending
image = np.zeros((720, 1280, 3), dtype=np.uint8)
face = np.ones((256, 256, 3), dtype=np.uint8) * 255
face_box = [100, 100, 356, 356]
mask_array = np.ones((614, 614, 3), dtype=np.uint8) * 255
crop_box = [100, 100, 714, 714]

try:
    res = get_image_blending(image, face, face_box, mask_array, crop_box)
    print("Success! Shape:", res.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
