from typing import Optional

import chex
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from src.tabs.environments.base_maenv import BaseMAEnv


class Visualizer:
    def __init__(
        self,
        env: BaseMAEnv,
        state_seq: chex.Array,
        reward_seq: chex.Array = None,
        interval: int = 120,
    ):
        self.env = env

        self.interval = interval
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))

    def animate(
        self,
        save_fname: Optional[str] = None,
        view: bool = True,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            init_func=self.init,
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)

        # Simply view it 3 times
        if view:
            plt.show(block=True)

    def init(self):
        self.im = self.env.init_render(self.ax, self.state_seq[0])

    def update(self, frame):
        self.im = self.env.update_render(self.im, self.state_seq[frame])
