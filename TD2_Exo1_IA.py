import numpy as np
import pandas as pd
import time
import sys

from mkl_random.mklrand import randint

UNIT = 80  # pixels
MAZE_H = 2  # 4  # grid height
MAZE_W = 6  # 4  # grid width
# SIZE = 30 #15
HALF_UNIT = UNIT / 2
HALF_UNIT_MINUS10 = HALF_UNIT - 10
SIZE = int(3 * UNIT / 8) # HALF_UNIT_MINUS10
NombreDeTourVoulu=30


if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['gauche', 'droite']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self.q_table = pd.DataFrame(columns=self.action_space, dtype=np.float64)
        self._build_maze()
        self.b = 0

        self.t1 = self.canvas.create_text(1*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="00")
        self.t2 = self.canvas.create_text(2*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="0.0")
        self.t3 = self.canvas.create_text(3*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="0.0")
        self.t4 = self.canvas.create_text(4*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="0.0")
        self.t5 = self.canvas.create_text(5*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="0.0")

        self.t1bis = self.canvas.create_text(1*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")
        self.t2bis = self.canvas.create_text(2*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")
        self.t3bis = self.canvas.create_text(3*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")
        self.t4bis = self.canvas.create_text(4*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")
        self.t5bis = self.canvas.create_text(5*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")

    def _build_maze(self):  # statique
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT ,
                                width=MAZE_W * UNIT )
        # create grids
        for c in range(MAZE_H + 1):
            x0, y0 = c * UNIT, 0
            x1, y1 = x0, y0 + MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
            self.canvas.create_line(x0 + UNIT, y0, x1 + UNIT, y1)
            self.canvas.create_line(x0 + UNIT * 2, y0, x1 + UNIT * 2, y1)
            self.canvas.create_line(x0 + UNIT * 3, y0, x1 + UNIT * 3, y1)
            self.canvas.create_line(x0 + UNIT * 4, y0, x1 + UNIT * 4, y1)

        for c in range(5):
            self.canvas.create_text(c * UNIT + UNIT / 3, MAZE_H * UNIT / 2 + 10, text="droite : ")

        for c in range(5):
            self.canvas.create_text(c * UNIT + UNIT / 3, MAZE_H * UNIT / 2 + 40, text="left : ")



        for r in range(MAZE_W + 1):
            x0, y0 = 0, r * UNIT
            x1, y1 = x0 + MAZE_W * UNIT, y0
            self.canvas.create_line(x0, y0, x1, y1)



        # creer le point de depart
        pointDepart = np.array([UNIT / 8, UNIT / 8])
        x0, y0 = pointDepart[0], pointDepart[1]
        self.rect = self.canvas.create_rectangle(x0, y0, x0 + 3 * UNIT / 4, y0 + 3 * UNIT / 4, fill='red')
        x0, y0 = pointDepart[0] + (MAZE_W - 1) * UNIT, pointDepart[1]
        x1, y1 = x0 + 3 * UNIT / 4, y0 + 3 * UNIT / 4
        self.oval = self.canvas.create_oval(x0, y0, x1, y1, fill="yellow")
        # pack all


        self.canvas.pack()  # affichage

    def step(self, action):  # dynamique
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 'droite':  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 'gauche':  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        E = str(self.canvas.coords(self.rect))
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        coor = str(self.canvas.coords(self.rect))

        if (self.canvas.coords(self.rect) == self.canvas.coords(self.oval)):
            print("stop")
            return 1
        else:
            return coor

    def check_state_exist(self):
        s = str(self.canvas.coords(self.rect))
        if s not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.action_space),
                    index=self.q_table.columns,
                    name=s,
                )
            )

    def apprendre(self, E, Ebis, action):

        self.check_state_exist()
        actionScores = self.q_table.loc[str(Ebis), :]

        if E == Ebis:
            return

        elif self.canvas.coords(self.rect) == self.canvas.coords(self.oval):
            r = 1

        else:
            r = 0.9 * np.max(actionScores)

        q_actuel = self.q_table.loc[str(E), action]
        self.q_table.loc[str(E), action]+= 0.1 * (r - q_actuel)

        if action=="droite":
            afficheDroite =  round(self.q_table.loc[str(E), action],4)
        else:
            afficheGauche = round(self.q_table.loc[str(E), action], 4)

        if E == [10,10,70,70] :

            if action == "droite":
                self.canvas.delete(self.t1)
                self.t1 = self.canvas.create_text(UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text=str(afficheDroite))
            else:
                self.canvas.delete(self.t1bis)
                self.t1bis = self.canvas.create_text(UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text=str(afficheGauche))

        if E == [90, 10, 150, 70]:
            if action == "droite":
               self.canvas.delete(self.t2)
               self.t2=self.canvas.create_text(2*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text=str(afficheDroite))
            else:
               self.canvas.delete(self.t2bis)
               self.t2bis = self.canvas.create_text(2*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text=str(afficheGauche))

        if E == [170, 10, 230, 70]:
            if action == "droite":
                self.canvas.delete(self.t3)
                self.t3=self.canvas.create_text(3*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text=str(afficheDroite))
            else:
                self.canvas.delete(self.t3bis)
                self.t3bis = self.canvas.create_text(3*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40,text=str(afficheGauche))

        if E == [250, 10, 310, 70]:
            if action == "droite":
                self.canvas.delete(self.t4)
                self.t4=self.canvas.create_text(4*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text=str(afficheDroite))
            else:
                self.canvas.delete(self.t4bis)
                self.t4bis = self.canvas.create_text(4*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text=str(afficheGauche))

        if E == [330, 10, 390, 70]:
            if action == "droite":
                self.canvas.delete(self.t5)
                self.t5=self.canvas.create_text(5*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text=str(afficheDroite))
            else:
                self.canvas.delete(self.t5bis)
                self.t5bis = self.canvas.create_text(5*UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text=str(afficheGauche))





    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        pointDepart = np.array([UNIT / 8, UNIT / 8])

        x0, y0 = pointDepart[0], pointDepart[1]
        x1, y1 = x0 + 3 * UNIT / 4, y0 + 3 * UNIT / 4
        self.rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill='red')

    def choisirAction(self, E):
        E = str(E)
        if E not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.action_space), index=["gauche", "droite"], name=E))

        if np.random.uniform() < 0.1:
            if np.random.uniform() < 0.5:
                return 'droite'
            else:
                return 'gauche'

        else:  # choisir la plus grosse stat

            s = str(self.canvas.coords(self.rect))
            actionScores = self.q_table.loc[s, :]  # [0.0,0.0] à l'indice s (s c'est le coor)
            return np.random.choice(actionScores[actionScores == np.max(
                actionScores)].index)  # returns le max entre ['gauche','droite'] --> la ça return le nom de la colonne


def update():
    E = env.canvas.coords(env.rect)  # état actuel
    action = env.choisirAction(E)

    a = env.step(str(action))

    Ebis = env.canvas.coords(env.rect)
    env.apprendre(E, Ebis, action)

    if a == 1:
        print(env.q_table)
        env.b += 1

        if env.b < NombreDeTourVoulu:
            env.reset()
            env.after(200, update)

    else:

        env.after(200, update)  # attendre en milliseconde et rapeller la fonction update


if __name__ == "__main__":
    env = Maze()
    update()
    env.mainloop()
