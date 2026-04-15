import numpy as np
from PIL import Image
import time
from musetalk.utils.blending import get_image_blending

# Test PIL conversion accuracy vs our numpy
image = np.ones((720, 1280, 3), dtype=np.uint8) * 100
face = np.ones((256, 256, 3), dtype=np.uint8) * 255

# Make a soft mask (gradient)
mask_array = np.zeros((614, 614, 3), dtype=np.uint8)
for i in range(614):
    mask_array[i, :, :] = int(i / 614 * 255)

face_box = [100, 100, 356, 356]
crop_box = [10, 10, 624, 624]

# Original PIL approach
def pil_blend():
    body = Image.fromarray(image[:,:,::-1])
    f = Image.fromarray(face[:,:,::-1])
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    face_large = body.crop(crop_box)
    mask_image = Image.fromarray(mask_array).convert("L")
    face_large.paste(f, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]

t0 = time.time()
res_pil = pil_blend()
t1 = time.time()
res_np = get_image_blending(image, face, face_box, mask_array, crop_box)
t2 = time.time()

print(f"PIL: {t1-t0:.3f}s, NP: {t2-t1:.3f}s")
diff = np.abs(res_pil.astype(int) - res_np.astype(int)).mean()
print(f"Average pixel difference: {diff:.3f}")
