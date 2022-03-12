import numpy as np
import matplotlib.pyplot as plt
def p(l,m,c,b,x,y):
    xs = m/c
    ys = l/b
    u = x - xs
    v = y - ys
    k = (u**2)/(b/(c*l))+(v**2)/(c/(b*m))
    print(xs)
    print(ys)
    return k,xs,ys
def main():
    l = 32.221
    b = 1.3847
    m = 27.432
    c = 0.5433
    x1 = 21
    y1 = 44
    x2 = 72
    y2 = 23
    k1,xs,ys = p(l,m,c,b,x1,y1)
    k2,xs,ys = p(l,m,c,b,x2,y2)
    u = np.linspace(-20,120,1000)
    v = np.linspace(-20,120,1000)
    u,v = np.meshgrid(u,v)
    q = ((u-xs)**2)/(b/(c*l))+((v-ys)**2)/(c/(b*m))
    plt.contour(u,v,q,[k1])
    plt.contour(u,v,q,[k2])
    plt.show()
if __name__ == "__main__":
    main()

#2   
import os
import math
import numpy as np
import matplotlib.pyplot as pyplot    
l = 32.221
b = 1.3847
m = 27.432
c = 0.5433
x = np.linspace(0,300,30)
y = np.linspace(0,110,21)
x,y = np.meshgrid (x,y)
dx = x*(l-b*y)
dy = y*(-m + c*x)
z = (dx**2 + dy**2)**(0.5)
dx = dx/z
dy = dy/z
pyplot.quiver(x,y,dx,dy)
pyplot.grid()
pyplot.title('Lotka-Volterra Vector Field')
x1 = 21
y1 = 44
x2 = 72
y2 = 23
x3 = 2
y3 = 16
a1 = l*math.log(y1) + m*math.log(x1)-b*y1 - c*x1
a2 = l*math.log(y2) + m*math.log(x2)-b*y2 - c*x2
a3 = l*math.log(y3) + m*math.log(x3)-b*y3 - c*x3
x4 = np.linspace(1,300,1000)
y4 = np.linspace(1,110,1000)
x4,y4 = np.meshgrid(x4,y4)
p = l*np.log(y4)+m*np.log(x4)-b*y4-c*x4
o1 = pyplot.contour(x4,y4,p,[a1],colors ='blue')
o2 = pyplot.contour(x4,y4,p,[a2],colors ='red')
o3 = pyplot.contour(x4,y4,p,[a3],colors ='green')
l1,_ = o1.legend_elements()
l2,_ = o2.legend_elements()
l3,_ = o3.legend_elements()
pyplot.legend([l1[0],l2[0],l3[0]],['(21,44)','(72,23)','(2,16)'], loc='upper right')
pyplot.title('Lotka-Volterra Orbits')
pyplot.xlabel('Hare')
pyplot.ylabel('Lynx')
pyplot.show()

#4
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def p():
    years=[]
    for i in range(1848,1908):
        years.append(i)
    return years
def q(z,t):
    l = 32.221
    b = 1.3847
    m = 27.432
    c = 0.5433
    x = z[0]
    y = z[1]
    dx = (x*(l-b*y))/59
    dy = (y*(-m+c*x))/59
    return [dx,dy]
hares = [21,12,24,50,80,80,90,69,80,93,72,27,14,16,38,5,153,145,106,46,23,2,4,8,7,60,46,50,103,87,68,17,10,17,16,15,46,55,137,137,95,37,22,50,54,65,60,81,95,56,18,5,2,15,2,6,45,50,58,20]
lynx = [44,20,9,5,5,6,11,23,32,34,23,15,7,4,5,5,16,36,77,68,37,16,8,5,7,11,19,31,43,27,18,15,9,8,8,27,52,74,79,34,19,12,8,9,13,20,37,56,39,27,15,4,6,9,19,36,59,61,39,10]
years=p()
grid = np.linspace(0,59,60)
plt.plot(grid,hares,'b',label='Hare')
plt.plot(grid,lynx,'g',label='Lynx')
plt.legend()
plt.show()
grid = np.linspace(0,59,1000)
xy = odeint(q,[hares[0],lynx[0]],grid)
plt.plot(grid,xy[:,0],'b',label ='Hare')
plt.plot(grid,xy[:,1],'g',label ='Lynx')
plt.legend()
plt.show()

#5
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def p(z,t): 
    l = 32.221
    b = 1.3847 
    m = 27.432 
    c = 0.5433
    x=z[0]
    y=z[1]
    dx = (x*(l - b*y))/59 
    dy = (y*(-m + c*x))/59 
    return [dx,dy]
hares = [21, 12, 24, 50, 80, 80, 90, 69, 80, 93, 72, 27, 14, 16, 38, 5, 153, 145, 106, 46, 23, 2, 4, 8, 7, 60, 46, 50, 103, 87,
68, 17, 10, 17, 16, 15, 46, 55, 137, 137, 95, 37, 22, 50, 54, 65, 60, 81, 95, 56, 18, 5, 2, 15, 2, 6, 45, 50, 58, 20]
lynx=[44, 20, 9, 5, 5, 6, 11, 23, 32, 34, 23, 15, 7, 4, 5, 5, 16, 36, 77, 68, 37, 16, 8, 5, 7, 11, 19, 31, 43, 27,
18, 15, 9, 8, 8, 27, 52, 74, 79, 34, 19, 12, 8, 9, 13, 20, 37, 56, 39, 27, 15, 4, 6, 9, 19, 36, 59, 61, 39, 10]
grid = np.linspace (0,59,1000)
xy1 = odeint(p , [0,lynx[0]] , grid) 
xy2 = odeint(p , [hares[0],0], grid)
plt.show()
plt.plot(grid, xy2[:,0], 'r',label='hares')
plt.plot(grid, xy2[:,1], 'g',label='lynx')
plt.legend() 
plt.show()

#7
from scipy.integrate import odeint 
import numpy as np
import matplotlib.pyplot as plt
def p(z,t): 
    u=z[0]
    v=z[1]
    b = 1.3847
    m = 27.432
    c = 0.5433 
    l = 32.221
    k = 400 
    q = 0.88
    x = (m/c)**(1/q)
    y = (l*((m/c)**(((1-q)/q))) - ((1/k)*((m/c)**(((2-q)/q)))))/b
    uu = l*(1-(2*x/k)) - b*q*(x**(q-1))*y 
    uv = -b*(x**q)
    vu = y*q*c*(x**(q-1))
    vv = c*(x**q) - m
    du = uu*u + uv*v
    dv = vu*u + vv*v
    return [du,dv]
u0 = -10 
v0 = 10
b = 1.3847
m = 27.432
c = 0.5433 
l = 32.221
k = 400 
q = 0.88
xs = (m/c)**(1/q)
ys = (l*((m/c)**(((1-q)/q))) - ((1/k)*((m/c)**(((2-q)/q)))))/b 
print("x^* = ")
print(xs)
print("y^* = ")
print(ys)
grid = np.linspace(0,3,1000)
xy = odeint(p,[u0,v0],grid)
plt.plot(xy[:,0] + xs, xy[:,1] + ys, label='Model 2')
plt.xlabel ("Hares") 
plt.ylabel ("Lynx") 
plt.legend ()
plt.show()

#8
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def sys_diff_eq ( z , t ) :
 l = 32.221
 b = 1.3847
 m = 27.432
 c = 0.5433
 x = z [ 0 ]
 y = z [ 1 ]
 dxdt = ( x *(l - b*y ))/59
 dydt = ( y* (-m + c *x ))/59
 return [ dxdt , dydt ]
hares = [21,12,24 , 50 , 80 , 80 , 90 , 69 , 80 , 93 , 72 ,27 , 14 , 16 , 38 , 5 , 153 , 145 , 106 , 46 , 23 , 2 , 4 , 8 , 7 , 60 , 46 , 50 , 103 , 87 ,68 , 17 , 10 , 17 , 16 , 15 , 46 , 55 , 137
         , 137 , 95 , 37 , 22 , 50 , 54 , 65 , 60 , 81 , 95 , 56 ,18 , 5 , 2 , 15 , 2 , 6 , 45 , 50 , 58 , 20]
lynx = [ 44 , 20 , 9 , 5 , 5 , 6 , 11 , 23 , 32 , 34 , 23 , 15 , 7 , 4 , 5 , 5 , 16 , 36 , 77 , 68 , 37 , 16 , 8 , 5 , 7 , 11 , 19 , 31 , 43 , 27 ,18 , 15 , 9 , 8 , 8 , 27 , 52 , 74 , 79 
        , 34 , 19 , 12 ,8 , 9 , 13 , 20 , 37 , 56 , 39 , 27 , 15 , 4 , 6 , 9 , 19 , 36 , 59 , 61 , 39 , 10]

t_grid = np . linspace ( 0 ,59 ,1000 )
xy_sol_1 = odeint(sys_diff_eq , [ 0 , lynx [0]] , t_grid )
xy_sol_2 = odeint(sys_diff_eq , [ hares [0] ,0] , t_grid )
plt.plot(t_grid , xy_sol_2 [: , 0 ] , ' r ' , label= 'hares' )
plt.plot(t_grid , xy_sol_2 [: , 1 ] , ' g ' , label= 'lynx' , linewidth =10)
plt.legend( )
plt.show( )

#9
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt 
def p():
    years=[]
    for i in range(1848,1908):
        years.append(i)
    return years
def q(z, t):
    x = z[0]
    y = z[1]
    b = 1.3847
    c = 0.5433
    l = 32.221
    m = 27.432
    K = 400
    q = 0.88
    dxdt = (l*x*(1 - (x/K))) - (b*(x**q)*y)
    dydt = y*(c*(x**q) - m)
    return [dxdt, dydt] 
hares = [21,12,24,50,80,80,90,69,80,93,72,27,14,16,38,5,153,145,106,46,23,2,4,8,7,60,46,50,103,87,68,17,10,17,16,15,46,55,137,137,95,37,22,50,54,65,60,81,95,56,18,5,2,15,2,6,45,50,58,20]
lynx = [44,20,9,5,5,6,11,23,32,34,23,15,7,4,5,5,16,36,77,68,37,16,8,5,7,11,19,31,43,27,18,15,9,8,8,27,52,74,79,34,19,12,8,9,13,20,37,56,39,27,15,4,6,9,19,36,59,61,39,10]
years=p()
x_0 = 21
y_0 = 44
xy = odeint(q, [x_0,y_0], grid)
grid = np.linspace(0,60,1000)
xy = odeint(q,[hares[0],lynx[0]],grid)
plt.plot(grid,xy[:,0],'b',label ='Hare')
plt.plot(grid,xy[:,1],'g',label ='Lynx')
plt.legend()
plt.show()
grid = np.linspace(0,59,60)
plt.plot(grid,hares,'b',label='Hare')
plt.plot(grid,lynx,'g',label='Lynx')
plt.legend()
plt.show()


#10
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt 
def eqn(z,t):
    u = z[0]
    v = z[1]
    b = 1.3847
    c = 0.5433
    l = 32.221
    m = 27.432
    K = 400
    q1 = 0.71
    q2 = 0.5
    x = (m/c)**(1/q)
    y = (l*((m/c)**((1-q)/q)) - ((1/K)*((m/c)**((2-q)/q))))/b
    u_u = l*(1-((2*x)/(K)))-(b*q*y*(x**(q-1)))
    u_v = (-b)*(x**q)
    v_u = c*y*q*(x**(q-1)) 
    v_v = c*(x**q) - m
    dudt = u_u*u + u_v*v
    dvdt = v_u*u + v_v*v
    return [dudt, dvdt]
u_0 = -10
v_0 = 10
b = 1.3847
c = 0.5433
l = 32.221
m = 27.432
K = 400
q0 = 0.88
q1 = 0.71
q2 = 0.5
x_star0 = (m/c)**(1/q0)
y_star0 = (l*((m/c)**((1-q0)/q0)) - ((1/K)*((m/c)**((2-q0)/q0))))/b
x_star1 = (m/c)**(1/q1)
y_star1 = (l*((m/c)**((1-q1)/q1)) - ((1/K)*((m/c)**((2-q1)/q1))))/b
x_star2 = (m/c)**(1/q2)
y_star2 = (l*((m/c)**((1-q2)/q2)) - ((1/K)*((m/c)**((2-q2)/q2))))/b
print("for q=0.88, x_star = ")
print(x_star0)
print("for q=0.88, y_star = ")
print(y_star0)
print("for q=0.71, x_star = ")
print(x_star1)
print("for q=0.71, y_star = ")
print(y_star1)
print("for q=0.5, x_star = ")
print(x_star2)
print("for q=0.5, y_star = ")
print(y_star2)
#plot
grid = np.linspace(0,3,1000)
xy = odeint(eqn, [u_0, v_0], grid)
plt.plot(xy[:,0] + x_star0, xy[:,1] + y_star0, label = "q = 0.88")
plt.xlabel("Hare")
plt.ylabel("Lynx")
plt.legend()
plt.show()
grid = np.linspace(0,3,1000)
xy = odeint(eqn, [u_0, v_0], grid)
plt.plot(xy[:,0] + x_star1, xy[:,1] + y_star1, label = "q = 0.71")
plt.xlabel("Hare")
plt.ylabel("Lynx")
plt.legend()
plt.show()
grid = np.linspace(0,3,1000)
xy = odeint(eqn, [u_0, v_0], grid)
plt.plot(xy[:,0] + x_star2, xy[:,1] + y_star2, label = "q = 0.5")
plt.xlabel("Hare")
plt.ylabel("Lynx")
plt.legend()
plt.show()