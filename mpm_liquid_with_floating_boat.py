import taichi as ti

ti.init(arch=ti.gpu)

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


n2_particles = 1024 
x2_ = ti.Vector.field(2, float, n2_particles)
x2 = ti.Vector.field(2, float, n2_particles)
x2c = ti.Vector.field(2, float, ())
v2 = ti.Vector.field(2, float, ())
rot = ti.field(float, ())
wrot = ti.field(float, ())


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass


    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * gravity
        dist = attractor_pos[None] - dx * ti.Vector([i, j])
        grid_v[i, j] += dist / (
            0.01 + dist.norm()) * attractor_strength[None] * dt * 10
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
            

    dis_mass = 0.0
    v2_fld = ti.Vector.zero(float, 2)
    J2 = 0.0
    for p in x2:
        Xp = x2[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_m = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            g_m = grid_m[base + offset]
            new_v += weight * g_v
            new_m += weight * g_m
        v2_fld += new_v
        J2 += (x2[p]-x2c[None]).cross(new_v)*new_m + (x2[p]-x2c[None]).cross(ti.Vector([0, -gravity * dt]))
        dis_mass += new_m
    v2_fld /= n2_particles
    v2_self = v2[None]
    v2_self.y -= dt * gravity
    inertia = n2_particles*0.8
    ratio_free = dis_mass/p_mass/inertia
    ratio = min(ratio_free, 1)
    v2[None] = v2_self*(1-ratio) + v2_fld*ratio
    floating = 20
    v2[None].y += ratio * ratio_free * dt * floating

    
    if (x2c[None].x-w2/2)*n_grid < bound and v2[None].x < 0:
        v2[None].x *= -1
    if (x2c[None].x+w2/2)*n_grid > n_grid - bound and v2[None].x > 0:
        v2[None].x *= -1
    if (x2c[None].y-h2/2)*n_grid < bound and v2[None].y < 0:
        v2[None].y *= -1
    if (x2c[None].y+h2/2)*n_grid > n_grid - bound and v2[None].y > 0:
        v2[None].y *= -1

    wrot[None] *= (1-ratio*wrot[None]**2*5e-4)
    wrot[None] += J2*0.4
    rot[None] += wrot[None]*dt
    c, s = ti.cos(rot[None]), ti.sin(rot[None])
    x2c[None] += dt * v2[None]
    for p in x2:
        x2[p].x = x2c[None].x +  c*x2_[p].x + s*x2_[p].y
        x2[p].y = x2c[None].y + -s*x2_[p].x + c*x2_[p].y
            
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * (g_v.outer_product(dpos) / dx**2)
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


w2, h2 = 0.2, 0.05 

@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.7 + 0.2, ti.random() * 0.7 + 0.2]
        v[i] = [0, -1]
        J[i] = 1

    x2c[None] = [0.4, 0.9]
    for i in range(n2_particles):
        x2_[i] = [ti.random()-0.5, ti.random()-0.5]
        x2_[i][0] += (x2_[i][1]+1) * x2_[i][0]
        x2_[i][0] = x2_[i][0]/2 *w2
        x2_[i][1] = x2_[i][1] *h2
        x2_[i]+= [0, 0.1*h2] #bias center
        x2[i] = x2_[i] + x2c[None]
    v2[None] = [0.0, 0.0]
    rot[None] = 20.0
    wrot[None] = 10.0
    

attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())

init()
gui = ti.GUI('floating boat')
# tick=0
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r': init()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]: break
    mouse = gui.get_cursor_pos()
    gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
    attractor_pos[None] = [mouse[0], mouse[1]]
    attractor_strength[None] = 0
    if gui.is_pressed(ti.GUI.LMB):
        attractor_strength[None] = 1
    if gui.is_pressed(ti.GUI.RMB):
        attractor_strength[None] = -1

    for s in range(50):
        substep()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.circles(x2.to_numpy(), radius=2.0, color=0xED553B)
    gui.show()
    # tick+=1
    # gui.show(f'img/{tick:0>3d}.png')
