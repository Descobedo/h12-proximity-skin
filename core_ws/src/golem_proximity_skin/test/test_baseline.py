import numpy as np
import pytest

from golem_proximity_skin.baseline import CAP_M, ENV, NONE, SELF, nearest_reading_clearance, zone_class


def test_zone_class_mapping():
    # generate.py: 0 none, 1 self, 2 static, 3 moving, 4 world
    assert zone_class(np.array([0, 1, 2, 3, 4])).tolist() == [NONE, SELF, ENV, ENV, ENV]


class _F:
    def __init__(self, readings, cls):
        self.readings, self.cls, self.n = readings, cls, readings.shape[0]


def test_nearest_reading_clearance_and_self_filter():
    r = np.zeros((1, 2, 64), np.uint16)
    c = np.zeros((1, 2, 64), np.uint8)
    r[0, 0, 5], c[0, 0, 5] = 40, SELF      # sensor 0 sees the robot at 4 cm
    r[0, 0, 6], c[0, 0, 6] = 300, ENV      # and an obstacle at 30 cm
    skin = np.array([0, 1])
    f = _F(r, c)
    assert np.allclose(nearest_reading_clearance(f, skin, 2, False)[0], [0.04, CAP_M])
    assert np.allclose(nearest_reading_clearance(f, skin, 2, True)[0], [0.30, CAP_M])


def test_model_shapes():
    torch = pytest.importorskip('torch')
    from golem_proximity_skin.baseline import build_model
    m = build_model(torch, n_sensors=5, n_joints=3, n_skins=2)
    logits, clear = m(torch.zeros(4, 5, 5, 8, 8), torch.zeros(4, 6))
    assert logits.shape == (4, 5, 3, 64) and clear.shape == (4, 2)
    assert float(clear.abs().max()) <= CAP_M


def test_zone_loss_lines_classes_up_with_zones():
    torch = pytest.importorskip('torch')
    from golem_proximity_skin.baseline import zone_loss
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 3, (4, 5, 64), generator=g)
    perfect = torch.nn.functional.one_hot(y, 3).permute(0, 1, 3, 2).float() * 20.0   # (B, S, 3, 64)
    ce = torch.nn.CrossEntropyLoss()
    assert float(zone_loss(ce, perfect, y)) < 1e-3
    assert float(zone_loss(ce, perfect, (y + 1) % 3)) > 5.0
