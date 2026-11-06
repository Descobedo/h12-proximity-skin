# H1-2 proximity skin

A Unitree **H1-2 humanoid wearing VL53L5CX time-of-flight proximity skins**, in
MuJoCo:
- HIRO-group's printed arm and knee skins, on the real robot model;
- a sensor model fitted from real recordings;
- an obstacle scene where the arm slows and stops for a person's hand;
- an ARMOR-style training-data generator, and a baseline perception model;
- tools for real HIRO-firmware skin boards;
- a patch to run the skins in the Correll Lab's GOLEM stack (ROS 2, RoboCasa).

```bash
python -m pip install -r requirements.txt
python setup_scene.py                                   # fetches the robot and skins, builds, tests (~2 min)
cd core_ws/src/golem_proximity_skin
python -m golem_proximity_skin.simulate --arm right --viewer
```

**For agents and anyone rebuilding the scene: start with [`AGENTS.md`](AGENTS.md).**
It covers the environment, every build step with its expected output, the
GOLEM integration, the known issues, and provenance. The package's own
[`README.md`](core_ws/src/golem_proximity_skin/README.md) documents each tool.

Third-party assets aren't stored here: the H1-2 model (Unitree, via
`correlllab/CL_Assets`) and the skins (`HIRO-group/proximity_point_motion`)
are fetched at pinned commits by `setup_scene.py`.
