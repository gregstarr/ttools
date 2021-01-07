import numpy as np
import matplotlib.pyplot as plt


class App:

    def __init__(self):
        self.t = 0
        self.fig, self.ax = self.create_fig()
        self.plot()
        cid = self.fig.canvas.mpl_connect('key_press_event', self.keypress)

    def create_fig(self):
        return plt.subplots()

    def plot(self):
        pass

    def clear(self):
        if isinstance(self.ax, np.ndarray):
            for a in self.ax.flatten():
                a.clear()
        else:
            self.ax.clear()

    def next(self):
        self.t += 1

    def prev(self):
        if self.t == 0:
            print("BEGINNING")
            return
        else:
            self.t -= 1

    # event handler
    def keypress(self, event):
        if event.key == 'right':
            self.next()
        elif event.key == 'left':
            self.prev()
        self.clear()
        self.plot()
        plt.draw()


if __name__ == "__main__":
    app = App()
    plt.show()
