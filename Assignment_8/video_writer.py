import av
import numpy as np
from tqdm import tqdm

fps = 30

container = av.open('test.mp4', mode='w')

STREAM_SIZE = 1024
stream = container.add_stream('mpeg4', rate=fps)
stream.width = STREAM_SIZE
stream.height = STREAM_SIZE
stream.pix_fmt = 'yuv420p'

N = 128
TILE_SIZE = STREAM_SIZE // N
assert STREAM_SIZE == TILE_SIZE * N
for frame_i in tqdm(range(512)):

    img = np.random.random((N, N, 1))
    img = np.round(255 * img).astype(np.uint8)
    img = np.clip(img, 0, 255)
    img = np.repeat(img, TILE_SIZE, axis=0)
    img = np.repeat(img, TILE_SIZE, axis=1)
    img = np.repeat(img, 3, axis=2)
    frame = av.VideoFrame.from_ndarray(img, format='rgb24')
    for packet in stream.encode(frame):
        container.mux(packet)

# Flush stream
for packet in stream.encode():
    container.mux(packet)

# Close the file
container.close()