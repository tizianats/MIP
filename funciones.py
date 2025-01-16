import numpy as np
import scipy.linalg
import pandas as pd 

def calcularLU(A):
    # Ingresa como parámetro la matriz que queremos descomponer
    L, U = [],[]

    m=A.shape[0] # filas
    n=A.shape[1] # columnas
    aPerm = A.copy()
    P = np.eye(n) # matriz permutacion

    p = np.arange(0, m) # vector permutacion

    if m!=n: # termina si la matriz no es cuadrada
        print('Matriz no cuadrada')
        return

    maximo = 0
    for k in range (0,n,1): # recorremos las columnas

        if aPerm[k,k] == 0: # nos fijamos si el coeficiente de la diagonal es 0
            for i in range(k+1,m): # recorremos las filas a partir del k
                if abs(aPerm[i][k]) == (np.max(abs(aPerm[k+1:, k]))): # buscamos el máximo en módulo
                    maximo = i # si i es la posición del máximo, se lo asignamos a "máximo"
                    p[[k, maximo]] = p[[maximo, k]] # permutamos
                    P = P[p]
                    aPerm = P@aPerm # matriz A permutada
                    break

        for i in range(k+1,m,1): # recorremos las filas a partir del k, independientemente de si aPerm[k,k]==0
            if aPerm[k+1:n,k].any() == 0: # Si encuentra un cero abajo de la diagonal incrementa el índice
                i += 1

            else:
                multiplicador = aPerm[i][k] / aPerm[k][k] # calcula el valor por el que tiene que multiplicar la fila, para poder eliminar el coeficiente
                aPerm[i][k] = multiplicador # reemplazamos el valor de la posición por el multiplicador (para ir formando L)
                for j in range(k+1,n): # recorremos el resto de la columna
                    aPerm[i,j] = aPerm[i, j] - multiplicador * aPerm[k, j] # hacemos la resta y reemplazamos las posiciones


    L = np.tril(aPerm,-1) + np.eye(aPerm.shape[0])
    U = np.triu(aPerm)

    return L, U, P

def inversaLU(L, U, P):
    n = L.shape[1] # columnas
    Inv = np.empty((n,n), float) # creamos una matriz con todos ceros
    I = np.eye(n) # matriz identidad de tamaño nxn

    for k in range(n): # recorremos las columnas
      y = scipy.linalg.solve_triangular(L, I[:, k], lower=True) # resolvemos el sistema Ly = I[:,k] donde I[:,k] representa el vector canónico de la columna k
      x = scipy.linalg.solve_triangular(U, y) # resolvemos el sistema Ux = y
      Inv[:,k] = x # asignamos x (vector) a la columna correspondiente en la matriz Inv

    Inv = Inv @ P # conseguimos la inversa de la matriz original al volver a multiplicar por la permutación hecha al hacer la descomoposición LU.

    return Inv


def MetodoPotencia(A):
  autovalores_iteraciones = []                      # Inicializamos la lista de autovalores por iteracion
  autovalores_promedio = 0                          # Variable para almacenar el promedio de los autovalores 
  eps = 0.0000001

  v= np.random.rand(A.shape[0])                     # Inicializamos el vector aleactorio
  v = v/np.linalg.norm(v)                           # Normalizamos el vector   

  autovalor= 0                                      # Variable para calcular la mejor aproximacion del autovalor de mayor modulo

  for k in range(100):                  
    v_actual = A @ v                                # Metodo de la potencia
    v_actual = v_actual/np.linalg.norm(v_actual)

    autovalor= (v_actual.T @ A @ v_actual)          # Calculo de coeficiente de Rayleigh    

    autovalores_iteraciones.append(autovalor)       # Guardamos la aproximacion del autovalor en cada iteracion 

    if(np.linalg.norm(v_actual - v) < eps):         # Condicion de corte
      break

    v= v_actual
    autovalores_promedio = np.mean(autovalores_iteraciones)     # Guardamos el promedio de los autovalores 
    desvio= pd.Series(autovalores_iteraciones).std()            # Tambien el desvio 

  return autovalor, autovalores_promedio, autovalores_iteraciones, v, desvio

def MonteCarlo(A):

  hist_autovalores= []          # Queremos repetir el procedimiento 250 veces y guardar los ultimos valores de cada repeticion

  for k in range(250):
    # repetimos el proceso con 250 vectores random

    autovalor, autovalores_promedio, autovalores_iteraciones, v, desvio= MetodoPotencia(A)

    hist_autovalores.append(autovalor)

  hist_autovalores= pd.Series(hist_autovalores)

  # Luego de guardar la mejor aproximacion en cada repeticion, devolvemos el promedio de todos y el desvio estandar

  return np.round(hist_autovalores.std(), 3), hist_autovalores.mean()