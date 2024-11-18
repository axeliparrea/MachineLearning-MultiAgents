import random
import numpy as np
import matplotlib.pyplot as plt

filas = 5
columnas = 5
energia_inicial = 10

class Robot:
    def __init__(self, energia, posicion_inicial):
        self.energia = energia
        self.posicion = posicion_inicial
        self.recompensa_acumulada = 0
        self.tiempo_total = 0

    def mover(self, nueva_posicion, habitacion):
        if nueva_posicion is None:
            return False
        if self.energia <= 0:
            return False
        fila, col = nueva_posicion
        recompensa = habitacion[fila, col]
        self.energia += recompensa - 1  # Coste de movimiento: -1
        self.recompensa_acumulada += recompensa
        habitacion[fila, col] = 0  # Recolecta Recom
        self.posicion = (fila, col)
        self.tiempo_total += 1
        if self.recompensa_acumulada < 0:
            return False
        return True

    def encontrar_vecinos(self):
        fila, col = self.posicion
        vecinos = []
        if fila > 0:
            vecinos.append((fila - 1, col))
        if fila < filas - 1:
            vecinos.append((fila + 1, col))
        if col > 0:
            vecinos.append((fila, col - 1))
        if col < columnas - 1:
            vecinos.append((fila, col + 1))
        return vecinos

    def moverse_aleatoriamente(self, habitacion):
        vecinos = self.encontrar_vecinos()
        return self.mover(random.choice(vecinos), habitacion)

    def moverse_hacia_max_recompensa(self, habitacion):
        mejor_celda = None
        mejor_recompensa = -float('inf')
        for fila in range(filas):
            for col in range(columnas):
                if habitacion[fila, col] > mejor_recompensa:
                    mejor_recompensa = habitacion[fila, col]
                    mejor_celda = (fila, col)
        return self.mover(mejor_celda, habitacion) if mejor_celda is not None else False

    def moverse_hacia_recompensa_mas_cercana(self, habitacion):
        mejor_celda = None
        menor_distancia = float('inf')
        fila, col = self.posicion
        for f in range(filas):
            for c in range(columnas):
                if habitacion[f, c] > 0:
                    distancia = abs(fila - f) + abs(col - c)
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        mejor_celda = (f, c)
        return self.mover(mejor_celda, habitacion) if mejor_celda is not None else False

# simulaciones
def ejecutar_simulacion(estrategia, energia_inicial, posicion_inicial):
    habitacion = np.random.randint(-10, 10, (filas, columnas))
    print(f"\nHabitación generada para la estrategia '{estrategia}':")
    print(habitacion)

    robot = Robot(energia=energia_inicial, posicion_inicial=posicion_inicial)
    while robot.energia > 0:
        if estrategia == "aleatoria":
            if not robot.moverse_aleatoriamente(habitacion):
                break
        elif estrategia == "max_recompensa":
            if not robot.moverse_hacia_max_recompensa(habitacion):
                break
        elif estrategia == "recompensa_cercana":
            if not robot.moverse_hacia_recompensa_mas_cercana(habitacion):
                break

    print(f"\nHabitación final después de la estrategia '{estrategia}':")
    print(habitacion)
    return robot

# posición inicial y energía
posicion_inicial = (0, 0)
estrategias = ["aleatoria", "max_recompensa", "recompensa_cercana"]
resultados = {}

for estrategia in estrategias:
    print(f"\nSimulación con la estrategia '{estrategia}':")
    robot = ejecutar_simulacion(estrategia, energia_inicial, posicion_inicial)
    resultados[estrategia] = {
        "recompensa_total": robot.recompensa_acumulada,
        "energia_restante": robot.energia,
        "tiempo_total": robot.tiempo_total
    }

# Resultados Terminal
for estrategia, resultado in resultados.items():
    print(f"\nEstrategia: {estrategia}")
    print(f"  Recompensa total acumulada: {resultado['recompensa_total']}")
    print(f"  Energía restante: {resultado['energia_restante']}")
    print(f"  Tiempo total (movimientos): {resultado['tiempo_total']}")

# Resultados
estrategia_names = list(resultados.keys())
recompensas = [resultados[e]["recompensa_total"] for e in estrategia_names]
energia_restante = [resultados[e]["energia_restante"] for e in estrategia_names]
tiempo_total = [resultados[e]["tiempo_total"] for e in estrategia_names]

fig, axs = plt.subplots(3, 1, figsize=(10, 8))
axs[0].bar(estrategia_names, recompensas, color='blue')
axs[0].set_title("Recompensa total acumulada por estrategia")
axs[1].bar(estrategia_names, energia_restante, color='green')
axs[1].set_title("Energía restante por estrategia")
axs[2].bar(estrategia_names, tiempo_total, color='red')
axs[2].set_title("Tiempo total (movimientos) por estrategia")

plt.tight_layout()
plt.show()
