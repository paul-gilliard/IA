import numpy as np
import pandas as pd
import time
import sys

from mkl_random.mklrand import randint

UNIT = 80  # pixels
MAZE_H = 8  # 4  # grid height
MAZE_W = 4  # 4  # grid width
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
        self.action_space = ['gauche', 'droite', 'haut', 'bas']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self.q_table = pd.DataFrame(columns=self.action_space, dtype=np.float64)
        self._build_maze()
        self.b = 0

#Emplacement score DROITE
        self.t1 = self.canvas.create_text(1 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="0.0")
        self.t2 = self.canvas.create_text(2 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="0.0")
        self.t3 = self.canvas.create_text(3 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="0.0")
        self.t4 = self.canvas.create_text(4 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 10, text="0.0")

        self.t5 = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H +2) * UNIT / 2 + 10, text="0.0")
        self.t6 = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H +2) * UNIT / 2 + 10, text="0.0")
        self.t7 = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H +2) * UNIT / 2 + 10, text="0.0")
        self.t8 = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H +2) * UNIT / 2 + 10, text="0.0")

        self.t9 = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 10, text="0.0")
        self.t10 = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 10, text="0.0")
        self.t11 = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 10, text="0.0")
        self.t12 = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 10, text="0.0")

        self.t13 = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 10, text="0.0")
        self.t14 = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 10, text="0.0")
        self.t15 = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 10, text="0.0")
        self.t16 = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 10, text="0.0")


# Emplacement score GAUCHE
        self.t1bis = self.canvas.create_text(1 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")
        self.t2bis = self.canvas.create_text(2 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")
        self.t3bis = self.canvas.create_text(3 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")
        self.t4bis = self.canvas.create_text(4 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 40, text="0.0")

        self.t5bis = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 40, text="0.0")
        self.t6bis = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 40, text="0.0")
        self.t7bis = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 40, text="0.0")
        self.t8bis = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 40, text="0.0")

        self.t9bis = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 40, text="0.0")
        self.t10bis = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 +40, text="0.0")
        self.t11bis = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 40, text="0.0")
        self.t12bis = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 40, text="0.0")

        self.t13bis = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 40, text="0.0")
        self.t14bis = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 40, text="0.0")
        self.t15bis = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 40, text="0.0")
        self.t16bis = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 40, text="0.0")


# Emplacement score HAUT

        self.t1haut = self.canvas.create_text(1 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 25, text="0.0")
        self.t2haut = self.canvas.create_text(2 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 25, text="0.0")
        self.t3haut = self.canvas.create_text(3 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 25, text="0.0")
        self.t4haut = self.canvas.create_text(4 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 25, text="0.0")

        self.t5haut = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 25, text="0.0")
        self.t6haut = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 25, text="0.0")
        self.t7haut = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 25, text="0.0")
        self.t8haut = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 25, text="0.0")

        self.t9haut = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 25, text="0.0")
        self.t10haut = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 25, text="0.0")
        self.t11haut = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 25, text="0.0")
        self.t12haut = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 25, text="0.0")

        self.t13haut = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 25, text="0.0")
        self.t14haut = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 25, text="0.0")
        self.t15haut = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 25, text="0.0")
        self.t16haut = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 25, text="0.0")

# Emplacement score BAS

        self.t1bas = self.canvas.create_text(1 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 55, text="0.0")
        self.t2bas = self.canvas.create_text(2 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 55, text="0.0")
        self.t3bas = self.canvas.create_text(3 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 55, text="0.0")
        self.t4bas = self.canvas.create_text(4 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 55, text="0.0")
        self.t5bas = self.canvas.create_text(5 * UNIT - UNIT / 4, MAZE_H * UNIT / 2 + 55, text="0.0")

        self.t5bas = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 55, text="0.0")
        self.t6bas = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 55, text="0.0")
        self.t7bas = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 55, text="0.0")
        self.t8bas = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 2) * UNIT / 2 + 55, text="0.0")

        self.t9bas = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 55, text="0.0")
        self.t10bas = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 55, text="0.0")
        self.t11bas = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 55, text="0.0")
        self.t12bas = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 4) * UNIT / 2 + 55, text="0.0")

        self.t13bas = self.canvas.create_text(1 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 55, text="0.0")
        self.t14bas = self.canvas.create_text(2 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 55, text="0.0")
        self.t15bas = self.canvas.create_text(3 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 55, text="0.0")
        self.t16bas = self.canvas.create_text(4 * UNIT - UNIT / 4, (MAZE_H + 6) * UNIT / 2 + 55, text="0.0")


    def _build_maze(self):  # statique
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT ,
                                width=MAZE_W * UNIT )
        # create grids
        for c in range(MAZE_H + 1):
            x0, y0 = c * UNIT, 0
            x1, y1 = x0, y0 + MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)


        for r in range(MAZE_W + 5):
            x0, y0 = 0, r * UNIT
            x1, y1 = x0 + MAZE_W * UNIT, y0
            self.canvas.create_line(x0, y0, x1, y1)

        for x in range (4) :

            for c in range(5):
                self.canvas.create_text(c * UNIT + UNIT / 3,  (MAZE_H + 2 * x)  * UNIT / 2 + 10, text="droite : ")

            for c in range(5):
                self.canvas.create_text(c * UNIT + UNIT / 3, (MAZE_H+ 2 * x) * UNIT / 2 + 40, text="left : ")

            for c in range(5):
                self.canvas.create_text(c * UNIT + UNIT / 3,  (MAZE_H + 2 * x)  * UNIT / 2 + 55, text="bas : ")

            for c in range(5):
                self.canvas.create_text(c * UNIT + UNIT / 3, (MAZE_H+ 2 * x) * UNIT / 2 + 25, text="haut : ")





        # creer le point de depart
        pointDepart = np.array([UNIT / 8, UNIT / 8])
        x0, y0 = pointDepart[0], pointDepart[1]
        self.rect = self.canvas.create_rectangle(x0, y0, x0 + 3 * UNIT / 4, y0 + 3 * UNIT / 4, fill='red')

        x0, y0 = pointDepart[0] + (MAZE_W - 2) * UNIT, pointDepart[1] + (MAZE_H - 7) * UNIT
        self.rectEnfer1 = self.canvas.create_rectangle(x0, y0, x0 + 3 * UNIT / 4, y0 + 3 * UNIT / 4, fill='yellow')

        x0, y0 = pointDepart[0] + (MAZE_W - 3) * UNIT, pointDepart[1] + (MAZE_H - 6) * UNIT
        self.rectEnfer2 = self.canvas.create_rectangle(x0, y0, x0 + 3 * UNIT / 4, y0 + 3 * UNIT / 4, fill='yellow')

        x0, y0 = pointDepart[0] + (MAZE_W - 2) * UNIT, pointDepart[1] + (MAZE_H-6) * UNIT
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
        elif action == 'haut':  # haut
            if s[1] > (MAZE_H - 5) * UNIT:
                base_action[1] -= UNIT
        elif action == 'bas':  # bas
            if s[1] <  3*UNIT:
                base_action[1] += UNIT

        E = str(self.canvas.coords(self.rect))
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        coor = str(self.canvas.coords(self.rect))

        if (self.canvas.coords(self.rect) == self.canvas.coords(self.oval)):
            return 1
        elif (self.canvas.coords(self.rect) == self.canvas.coords(self.rectEnfer1)):
            return 1
        elif (self.canvas.coords(self.rect) == self.canvas.coords(self.rectEnfer2)):
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

        elif self.canvas.coords(self.rect) == self.canvas.coords(self.rectEnfer1):
            r = -1

        elif self.canvas.coords(self.rect) == self.canvas.coords(self.rectEnfer2):
            r = -1

        else:
            r = 0.9 * np.max(actionScores)

        q_actuel = self.q_table.loc[str(E), action]
        self.q_table.loc[str(E), action]+= 0.1 * (r - q_actuel)



        print(action)
        if action=="droite":
            afficheDroite =  round(self.q_table.loc[str(E), action],4)
        elif action =="gauche":
            afficheGauche = round(self.q_table.loc[str(E), action], 4)
        elif action == "haut":
            afficheHaut = round(self.q_table.loc[str(E), action], 4)
        else :
            afficheBas = round(self.q_table.loc[str(E), action], 4)




        if E == [10, 10, 70, 70]:

            if action == "droite":
                coord = self.canvas.coords(self.t1)
                print(coord)
                self.canvas.delete(self.t1)
                self.t1 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action =="gauche":
                coord = self.canvas.coords(self.t1bis)
                self.canvas.delete(self.t1bis)
                self.t1bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action=="haut":
                coord = self.canvas.coords(self.t1haut)
                self.canvas.delete(self.t1haut)
                self.t1haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t1bas)
                self.canvas.delete(self.t1bas)
                self.t1bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))



        if E == [90, 10, 150, 70]:
            if action == "droite":
                coord = self.canvas.coords(self.t2)
                print(coord)
                self.canvas.delete(self.t2)
                self.t2 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t2bis)
                self.canvas.delete(self.t2bis)
                self.t2bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t2haut)
                self.canvas.delete(self.t2haut)
                self.t2haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t2bas)
                self.canvas.delete(self.t2bas)
                self.t2bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))



        if E == [170, 10, 230, 70]:
            if action == "droite":
                coord = self.canvas.coords(self.t3)
                print(coord)
                self.canvas.delete(self.t3)
                self.t3 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t3bis)
                self.canvas.delete(self.t3bis)
                self.t3bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t3haut)
                self.canvas.delete(self.t3haut)
                self.t3haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t3bas)
                self.canvas.delete(self.t3bas)
                self.t3bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [250, 10, 310, 70]:
            if action == "droite":
                coord = self.canvas.coords(self.t4)
                print(coord)
                self.canvas.delete(self.t4)
                self.t4 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t4bis)
                self.canvas.delete(self.t4bis)
                self.t4bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t4haut)
                self.canvas.delete(self.t4haut)
                self.t4haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t4bas)
                self.canvas.delete(self.t4bas)
                self.t4bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [10, 90, 70, 150]:
            if action == "droite":
                coord = self.canvas.coords(self.t5)
                print(coord)
                self.canvas.delete(self.t5)
                self.t5 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t5bis)
                self.canvas.delete(self.t5bis)
                self.t5bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t5haut)
                self.canvas.delete(self.t5haut)
                self.t5haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t5bas)
                self.canvas.delete(self.t5bas)
                self.t5bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))


        if E == [90, 90, 150, 150]:
            if action == "droite":
                coord = self.canvas.coords(self.t6)
                print(coord)
                self.canvas.delete(self.t6)
                self.t6 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t6bis)
                self.canvas.delete(self.t6bis)
                self.t6bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t6haut)
                self.canvas.delete(self.t6haut)
                self.t6haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t6bas)
                self.canvas.delete(self.t6bas)
                self.t6bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [170, 90, 230, 150]:
            if action == "droite":
                coord = self.canvas.coords(self.t7)
                print(coord)
                self.canvas.delete(self.t7)
                self.t7 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t7bis)
                self.canvas.delete(self.t7bis)
                self.t7bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t7haut)
                self.canvas.delete(self.t7haut)
                self.t7haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t7bas)
                self.canvas.delete(self.t7bas)
                self.t7bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [250, 90, 310, 150]:
            if action == "droite":
                coord = self.canvas.coords(self.t8)
                print(coord)
                self.canvas.delete(self.t8)
                self.t8 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t8bis)
                self.canvas.delete(self.t8bis)
                self.t8bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t8haut)
                self.canvas.delete(self.t8haut)
                self.t8haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t8bas)
                self.canvas.delete(self.t8bas)
                self.t8bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [10, 170, 70, 230]:
            if action == "droite":
                coord = self.canvas.coords(self.t9)
                print(coord)
                self.canvas.delete(self.t9)
                self.t9 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t9bis)
                self.canvas.delete(self.t9bis)
                self.t9bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t9haut)
                self.canvas.delete(self.t9haut)
                self.t9haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t9bas)
                self.canvas.delete(self.t9bas)
                self.t9bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [90, 170, 150, 230]:
            if action == "droite":
                coord = self.canvas.coords(self.t10)
                print(coord)
                self.canvas.delete(self.t10)
                self.t10 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t10bis)
                self.canvas.delete(self.t10bis)
                self.t10bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t10haut)
                self.canvas.delete(self.t10haut)
                self.t10haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t10bas)
                self.canvas.delete(self.t10bas)
                self.t10bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [170, 170, 230, 230]:
            if action == "droite":
                coord = self.canvas.coords(self.t11)
                print(coord)
                self.canvas.delete(self.t11)
                self.t11 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t11bis)
                self.canvas.delete(self.t11bis)
                self.t11bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t11haut)
                self.canvas.delete(self.t11haut)
                self.t11haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t11bas)
                self.canvas.delete(self.t11bas)
                self.t11bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [250, 170, 310, 230]:
            if action == "droite":
                coord = self.canvas.coords(self.t12)
                print(coord)
                self.canvas.delete(self.t12)
                self.t12 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t12bis)
                self.canvas.delete(self.t12bis)
                self.t12bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t12haut)
                self.canvas.delete(self.t12haut)
                self.t12haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t12bas)
                self.canvas.delete(self.t12bas)
                self.t12bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [10, 250, 70, 310]:
            if action == "droite":
                coord = self.canvas.coords(self.t13)
                print(coord)
                self.canvas.delete(self.t13)
                self.t13 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t13bis)
                self.canvas.delete(self.t13bis)
                self.t13bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t13haut)
                self.canvas.delete(self.t13haut)
                self.t13haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t13bas)
                self.canvas.delete(self.t13bas)
                self.t13bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [90, 250, 150, 310]:
            if action == "droite":
                coord = self.canvas.coords(self.t14)
                print(coord)
                self.canvas.delete(self.t14)
                self.t14 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t14bis)
                self.canvas.delete(self.t14bis)
                self.t14bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t14haut)
                self.canvas.delete(self.t14haut)
                self.t14haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t14bas)
                self.canvas.delete(self.t14bas)
                self.t14bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [170, 250, 230, 310]:
            if action == "droite":
                coord = self.canvas.coords(self.t15)
                print(coord)
                self.canvas.delete(self.t15)
                self.t15 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t15bis)
                self.canvas.delete(self.t15bis)
                self.t15bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t15haut)
                self.canvas.delete(self.t15haut)
                self.t15haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t15bas)
                self.canvas.delete(self.t15bas)
                self.t15bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))

        if E == [250, 250, 310, 310]:
            if action == "droite":
                coord = self.canvas.coords(self.t16)
                print(coord)
                self.canvas.delete(self.t16)
                self.t16 = self.canvas.create_text(coord[0], coord[1], text=str(afficheDroite))
            elif action == "gauche":
                coord = self.canvas.coords(self.t16bis)
                self.canvas.delete(self.t16bis)
                self.t16bis = self.canvas.create_text(coord[0], coord[1], text=str(afficheGauche))
            elif action == "haut":
                coord = self.canvas.coords(self.t16haut)
                self.canvas.delete(self.t16haut)
                self.t16haut = self.canvas.create_text(coord[0], coord[1], text=str(afficheHaut))
            else:
                coord = self.canvas.coords(self.t16bas)
                self.canvas.delete(self.t16bas)
                self.t16bas = self.canvas.create_text(coord[0], coord[1], text=str(afficheBas))




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
        random=np.random.uniform()
        if E not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.action_space), index=["gauche", "droite","haut","bas"], name=E))

        if np.random.uniform() < 0.1:
            if random < 0.25:
                return 'droite'
            elif random>=0.25 and random < 0.5 :
                return 'gauche'
            elif random>0.75 :
                return 'haut'
            else:
                return "bas"

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
