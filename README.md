# toy-allegro-pol

Minimal toy project for understanding the `allegro-pol` program architecture with a fully controlled generalized potential:

```text
U(R, E) -> energy
       -> -dU/dR = forces
       -> -dU/dE = response
```

Current stage:

- a small analytic toy physics model
- a dataset generation script
- a simple config layout that can grow into training and evaluation later

Next planned stages:

- learn `U(R, E)` with a neural model
- train on energy, forces, and response jointly
- verify autograd against finite differences
