import os
import socket
import time

import numpy as np
import pytest

from golem_proximity_skin import vl53l5cx
from golem_proximity_skin.hiro_udp import (
    FIRMWARE_NO_TARGET_MM, FULL_PACKET, SHORT_PACKET, Receiver, Recorder,
    encode_packet, load_boards, parse_packet)
from golem_proximity_skin.layout import load_layout

PKG = os.path.join(os.path.dirname(__file__), '..')
LAYOUT = os.path.join(PKG, 'config', 'h12_skin.yaml')
BOARDS = os.path.join(PKG, 'config', 'hiro_boards.yaml')


def _frame():
    r = np.arange(64, dtype=np.uint16) * 50 + 30
    r[[3, 40]] = vl53l5cx.NO_TARGET
    return r


def test_packet_matches_the_firmware_layout():
    r = _frame()
    pkt = encode_packet(2, r)
    assert len(pkt) == FULL_PACKET == 257
    assert pkt[0] == 2
    filt = np.frombuffer(pkt, '<u2', 64, 1)
    assert filt[3] == FIRMWARE_NO_TARGET_MM and filt[5] == r[5]
    i, readings, raw = parse_packet(pkt)
    assert i == 2 and np.array_equal(readings, r) and raw is not None


def test_short_and_out_of_range_packets():
    r = _frame()
    i, readings, raw = parse_packet(encode_packet(1, r)[:SHORT_PACKET])
    assert raw is None and np.array_equal(readings, r)
    assert parse_packet(b'\x00' * 10) is None
    over = bytes([0]) + np.full(64, 4500, '<u2').tobytes()
    assert (parse_packet(over)[1] == vl53l5cx.NO_TARGET).all()


def test_board_map_loads_and_checks():
    layout = load_layout(LAYOUT)
    boards = load_boards(BOARDS, layout)
    arm = next(b for b in boards if b.skin == 'right_upper_arm')
    assert arm.port == 5007 and arm.layout_sensor(0) == arm.sensors[0]
    assert arm.layout_sensor(len(arm.sensors)) == -1


def test_board_map_rejects_bad_sensors(tmp_path):
    layout = load_layout(LAYOUT)
    p = tmp_path / 'b.yaml'
    p.write_text('boards:\n  - skin: right_knee\n    sensors: [0, 0]\n')
    with pytest.raises(ValueError):
        load_boards(str(p), layout)
    p.write_text('boards:\n  - skin: right_knee\n    sensors: [9]\n')
    with pytest.raises(ValueError):
        load_boards(str(p), layout)


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_receive_and_record_round_trip(tmp_path):
    layout = load_layout(LAYOUT)
    board = load_boards(BOARDS, layout)[0]
    board.port = _free_port()
    rx = Receiver([board], iface='127.0.0.1', multicast=False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    frames = []
    try:
        for k in range(len(board.sensors)):
            tx.sendto(encode_packet(k, _frame()), ('127.0.0.1', board.port))
        deadline = time.monotonic() + 2.0
        while len(frames) < len(board.sensors) and time.monotonic() < deadline:
            frames += rx.poll(0.1)
    finally:
        tx.close()
        rx.close()
    assert sorted(f.index for f in frames) == list(range(len(board.sensors)))
    rec = Recorder(str(tmp_path / 'r.npz'), [board], LAYOUT, BOARDS)
    for f in frames:
        rec.add(f)
    d = np.load(rec.save())
    assert d['packets'].shape == (len(frames), FULL_PACKET)
    assert np.array_equal(d['readings_mm'][0], _frame())
    assert (d['layout_sensors'] == [board.layout_sensor(i) for i in d['skin_indices']]).all()


def _boards_file(tmp_path, extra=''):
    p = tmp_path / 'boards.yaml'
    p.write_text('# a comment that must survive\nboards:\n  - skin: right_upper_arm\n'
                 '    sensors: [5, 4, 3, 0]\n' + extra + '    verified: true\n')
    return str(p)


def test_masked_zones_load_and_are_checked(tmp_path):
    layout = load_layout(LAYOUT)
    b = load_boards(_boards_file(tmp_path, '    masked_zones: {1: [0, 1, 9]}\n'), layout)[0]
    assert b.masked_zones == {1: [0, 1, 9]} and b.zone_mask(1)[[0, 1, 9]].all() and not b.zone_mask(0).any()
    with pytest.raises(ValueError):
        load_boards(_boards_file(tmp_path, '    masked_zones: {7: [0]}\n'), layout)     # no firmware 7
    with pytest.raises(ValueError):
        load_boards(_boards_file(tmp_path, '    masked_zones: {1: [64]}\n'), layout)    # no zone 64


def test_receiver_masks_zones(tmp_path):
    layout = load_layout(LAYOUT)
    board = load_boards(_boards_file(tmp_path, '    masked_zones: {1: [0, 63]}\n'), layout)[0]
    board.port = _free_port()
    rx = Receiver([board], iface='127.0.0.1', multicast=False)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        tx.sendto(encode_packet(1, _frame()), ('127.0.0.1', board.port))
        frames, deadline = [], time.monotonic() + 2.0
        while not frames and time.monotonic() < deadline:
            frames = rx.poll(0.1)
    finally:
        tx.close()
        rx.close()
    f = frames[0]
    assert f.readings[0] == 0 and f.readings[63] == 0 and f.raw[0] == 0 and f.readings[5] == _frame()[5]
    assert parse_packet(f.packet)[1][0] == _frame()[0]          # the packet itself is kept as received


def test_measure_and_write_masks(tmp_path):
    from golem_proximity_skin.hiro_udp import measure_masks, write_masks
    rng = np.random.default_rng(0)
    n = 200
    raw = rng.uniform(1500, 2500, (n, 64))
    raw[:, [0, 1, 8]] = 3.0                                        # three zones see the shell
    filt = raw.copy()
    pk = np.zeros((n, 257), np.uint8)
    for i in range(n):
        pk[i] = np.frombuffer(bytes([1]) + filt[i].astype('<u2').tobytes() + raw[i].astype('<u2').tobytes(), np.uint8)
    rec = tmp_path / 'open.npz'
    np.savez(rec, packets=pk, ports=np.full(n, 5007, np.uint16), skin_indices=np.ones(n, np.uint8))
    zones, st = measure_masks(str(rec), margin=0)[(5007, 1)]
    assert zones == [0, 1, 8] and st['obscured'] == 3
    grown, _ = measure_masks(str(rec), margin=1)[(5007, 1)]
    assert set(grown) == {0, 1, 2, 8, 9, 16}
    path = _boards_file(tmp_path)
    write_masks(path, 'right_upper_arm', {1: zones})
    text = open(path).read()
    assert text.startswith('# a comment that must survive') and 'masked_zones: {1: [0, 1, 8]}' in text
    write_masks(path, 'right_upper_arm', {1: [5]})                 # rewriting replaces, never duplicates
    assert open(path).read().count('masked_zones') == 1
    assert load_boards(path, load_layout(LAYOUT))[0].masked_zones == {1: [5]}
