# %%
import numpy as np
# %%
sine_angles = []
cosine_angles = []
folder = Path("../numpyarrays/")
for i in range(1, 112):
    # print(i)
    obs = np.load(f"/home/garlan/git/missle-octo-sniffle/numpyarrays/100_{i}.npy")
    obs.shape
    missle_positions = obs[0]
    # print(missle_positions.shape)
    sine_angle = obs[2,0,0,0]
    cosine_angle = obs[2,0,1,0]
    sine_angles.append(sine_angle)
    cosine_angles.append(cosine_angle)
# %%
import matplotlib.pyplot as plt
# %%
sine_angles = np.array(sine_angles)
print(f"sine_angles.shape: {sine_angles.shape}")
# %%

plt.scatter( np.arange(sine_angles.shape[0]) ,  sine_angles)
plt.scatter( np.arange(sine_angles.shape[0]) ,  cosine_angles)
# %%
# ffmpeg -i images/500/500_%d.png  myvideo500.mp4