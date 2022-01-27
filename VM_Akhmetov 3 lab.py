import numpy as np
import matplotlib.pyplot as plt
import time
import math

#-------------------------------------------------
#-Задача о растяжении стержня переменного сечения-
#-Параметры задачи--------------------------------
#-------------------------------------------------

L = 5 # длина балки
g = 9.81 # ускорение свободного падения
P_const = 30000 # сила
E = 30 * (10 ** 9) # модуль упругости
ro = 2150 # плотность
b = 0.1 # ширина балки

#-------------------------------------------------

def h(x):
  return 0.1 + x / 10 / L

def A(x):
  return b * h(x)

def C(x):
  return E * A(x)

def f(x):
  return ro * g * A(x) 

#--------------------------------------------------
#-Точное решение-----------------------------------
#--------------------------------------------------
def real_solve(x):
  k1 = - (5 * P_const + L * ro * g * b) * 4 * L / E / b
  k2 = - 0.5 * k1 * math.log(L)
  return x * 0.5 * L * ro * g / E + 0.5 * k1 * math.log(L + x) + x * x * 0.25 * ro * g / E + k2
  
  # c1 = (-P_const - 0.15 * b * g * L * ro) / (b * E)
  # c2 = (L * (10 * P_const + 2 * b * g * L * ro) * np.log(L)) / (b * E)
  # a = c2 + (g * ro * (0.5 * L + 0.25 * x) * x + L * (-0.5 * g * L * ro + 10 * c1 * E) * np.log(L + x)) / E
  # return a
  # k1 = (- P - 0.15 * b * g * L * ro) / (b * E)
  # k2 = (L * (10. * P + 21q. * b * g * L * ro) * math.log(L)) / (b * E)
  # return k2 + (g * ro * (0.5 * L + 0.25 * x) * x + L * (- 0.5 * g * L * ro + 10. * k1 * E) * math.log(L + x)) / E

#---------------------------------------------------

# нахождение параметров для метода Гивенса
def givens_parameters(a, b):
  if b == 0:
    return 1, 0
  if abs(b) > abs(a):
    if a == 0:
      r = 0
    else:
      r = - a / b
    s = 1 / (math.sqrt(1 + r ** 2))
    c = s * r
    return c, s
  else:
    r = -b / a
    c = 1 / (math.sqrt(1 + r ** 2))
    s = c * r
    return c, s

# Метод вращение Гивенса
def givens_matrix(A, b, m, n):
  for i in range(n):
    for j in range(i, m - 1):
      # высчитываем cos, sin
      c_g, s_g = givens_parameters(A[i, i], A[j + 1, i])
      r1 = np.copy(A[i, i:n])
      r2 = np.copy(A[j + 1, i:n])
      A[i, i:n] = c_g * r1 - s_g * r2
      A[j + 1, i:n] = s_g * r1 + c_g * r2
      # преобразование правой части
      r3 = np.copy(b[i])
      r4 = np.copy(b[j + 1])
      b[i] = c_g * r3 - s_g * r4
      b[j + 1] = s_g * r3 + c_g * r4
  return A, b

#-------------------------------------------------

def psi(i, x):
  if i in range(1, N):
    if (x >= setka[i - 1]) and (x <= setka[i]):
      return (x - setka[i - 1]) / (setka[i] - setka[i - 1])
    elif (x >= setka[i]) and (x <= setka[i + 1]):
      return (setka[i + 1] - x) / (setka[i + 1] - setka[i])
    else:
      return 0
  if i == 0:
    if (x >= setka[0]) and (x <= setka[1]):
      return (setka[1] - x) / (setka[1] - setka[0])
    else:
      return 0
  if i == N:
    if (x >= setka[N - 1]) and (x <= setka[N]):
      return (x - setka[N - 1]) / (setka[N] - setka[N - 1])
    else:
      return 0

#-------------------------------------------------

def K_matrix(setka, N):
  L_e = setka[1] - setka[0]
  K1 = np.zeros((N + 1, N + 1))

  K_loc = np.array([[1, -1], [-1, 1]])

  for i in range(N):
    K_step = np.zeros((N + 1, N + 1))
    K_step[i:i + 2, i:i + 2] = K_loc
    K1 += K_step

  #print('K1 =', K1)

  K2 = np.zeros((N + 1, N + 1))

  for i in range(N):
    K_step = np.zeros((N + 1, N + 1))
    K_step[i:i + 2, i:i + 2] = (setka[i + 1] + setka[i]) * K_loc 
    K2 += K_step
  
  K2 *= 0.5 / L
  #print('K2 =', K2)  

  K3 = np.zeros((N + 1, N + 1))

  K3 = (K1 + K2) * E * b / 10 / L_e
  K3[0][1] = 0
  #K3[-1][-2] = 0
  print('Обусловленность матрицы: %.2e' % np.linalg.cond(K3))
  #print('K =', K3)
  return K3

#-------------------------------------------------

def P_matrix(setka, N):
  L_e = setka[1] - setka[0]
  P = np.zeros((N + 1, N + 1))
  P_loc = np.array([[2, 1], [1, 2]])

  for i in range(N):
    P_step = np.zeros((N + 1, N + 1))
    P_step[i:i + 2, i:i + 2] = P_loc
    P += P_step
  P[0][0] = 0
  P[0][1] = 0
  #P[-1][-2] = 0
  
  P *= - L_e / 6
  #print('P =', P)
  return P

#------------------------------------------------- 

def solver(setka, N, K, P):
  f_x = np.zeros(N + 1)
  for i in range(N + 1):
    f_x[i] = f(setka[i])

  b_rightpart = P.dot(f_x)
  #b_rightpart = np.dot(P, f_x)
  #print('f =', b_rightpart)
  b_rightpart[-1] += - P_const 
  #print('b =', b_rightpart)

  # вектор решения
  u = np.zeros(N + 1)
  u = np.linalg.solve(K, b_rightpart)
  #print('u = ', u)
  # K_new, b_rightpart_new = givens_matrix(K, b_rightpart, N + 1, N + 1)

  # # обратный ход Гаусса для нахождения коэффициентов c_i
  # u[-1] = b_rightpart_new[-1] / K_new[-1][-1]
  # for i in range(N - 1, -1, -1):
  #   sum_G = 0
  #   for j in range(i + 1, N+1, 1):
  #     sum_G += + K_new[i][j] * u[j]
  #   u[i] = (b_rightpart_new[i] - sum_G) / K_new[i][i]
  #print('Обусловленность матрицы: %.2e' % np.linalg.cond(K, 2))
  # print('u =', u)

  n_uzlov = 1000
  x = np.linspace(0, L, n_uzlov)
  u_alg = np.zeros(n_uzlov)
  # точное решение
  u_exact = []
  for i in range(n_uzlov):
    u_exact.append(real_solve(x[i]))

  # приближенное решение
  for i in range(n_uzlov):
    for j in range(N + 1):
      u_alg[i] += psi(j, x[i]) * u[j]

  plt.figure(figsize=(8, 8))
  plt.plot(x, u_exact, linewidth = 3, label = 'Точное решение')
  plt.plot(x, u_alg, linewidth = 2, linestyle = '--', label = 'Приближенное решение при N='+str(N))
  plt.legend()
  plt.grid()
  plt.show()  
  return u_exact, u_alg 


#-------------------------------------------------
def error(u_ex, u_h, err, count):
  n_uzlov = 1000
  x = np.linspace(0, L, n_uzlov)
  # Рассчет относительной и абсолютной погрешностей
  E_r = 0
  E_a = 0
  #print('len =', len(W_graf))
  for jj in range(1000):
    #print(len(W_graf[jj * (Q + 1) + 1 : (jj + 1) * (Q + 1)]))

    M1 = abs(u_ex[jj] - u_h[jj])
    M2 = abs(u_ex[jj])
    if M1 > E_a:
      E_a = M1
    if M2 > E_r:
      E_r = M2

  E_r = E_a / E_r
  #---------------------------------
  E1 = 0
  E2 = 0
  for i in range(n_uzlov - 1):
    E1 += ((u_ex[i] - u_h[i]) ** 2 + (u_ex[i + 1] - u_h[i + 1]) ** 2) * (x[i + 1] - x[i]) / 2
    E2 += (u_ex[i] ** 2 + u_ex[i + 1] ** 2) * (x[i + 1] - x[i]) / 2

  # f1_0 = (u_ex[0] - u_h[0]) ** 2
  # f1_L = (u_ex[-1] - u_h[-1]) ** 2
  # E_1 = (f1_0 + f1_L) * L / 2

  # f2_0 = u_ex[0] ** 2
  # f2_L = u_ex[-1] ** 2
  # E_2 = (f2_0 + f2_L) * L / 2

  E_l = (E1 / E2) ** 0.5
  #---------------------------------
  print('E_r =', f'{E_r:.2e}')  
  print('E_l =', f'{E_l:.2e}')  

  err[0, count] = E_r
  err[1, count] = E_l

  if N > 5:
    print('Порядок для погрешности в норме: %.3e' % np.log2(err[1, count - 1] / err[1, count]), end='\n')
    print('Порядок для относительной погрешности: %.3e' % np.log2(err[0, count - 1] / err[0, count]), end='\n')

#-------------------------------------------------
N = 5 # количество отрезков
err = np.zeros((2, 9))
count = 0
while N <= 1280:
#   t0 = time.clock()
  setka = np.linspace(0, L, N + 1)
  K = K_matrix(setka, N)
  P = P_matrix(setka, N) 
#   t1 = time.clock()
#   print("Time elapsed: ", f'{t1 - t0:.3f}') 
  u_ex, u_h = solver(setka, N, K, P)
  error(u_ex, u_h, err, count)
  N *= 2
  count += 1
  print()
