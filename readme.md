# Fun game

* can i train an RL to shoot down missles?

## Plan

* make a simple pygame
  * missles come across the screen
  * you have to shoot them down
  * if one gets past, then you loose
  * your score = time till the end.
* then use some RL library to
  * play the game.

```bash
conda activate missle_defense
# RUN the RL game on a cuda enable computer with . 
python src/missle_agent/missle_agent_skrl.py 
```

Run the game as a human with.

```bash
conda activate missle_defense
python .\src\missle_defense\missle_game.py
```

* use the keyboard arrow keys to turn your gun
* use the space bar to shoot
```bash
ffmpeg -i images/100/100_%d.png  myvideo100.mp4

python utils/make_videos_from_saved_obs.py  

```
look at tensor board. 
```
tensorboard --logdir ./missle_ddqn/
```