"""Tests for numericvision.test_detect_box_sequences()."""
import os.path
import pytest
from numericvision import detect_box_sequences


HERE = os.path.dirname(__file__)


#
# Tests
#
def test_detect_box_sequences_1():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/01.jpg")}
    assert (
        (1571, 428) in keys_to_sequences
        and (1586, 598) in keys_to_sequences
        and (902, 2525) in keys_to_sequences
        and (1330, 2417) in keys_to_sequences
        and (1718, 2315) in keys_to_sequences
    )


def test_detect_box_sequences_2():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/02.jpg")}
    assert (
        (703, 2591) in keys_to_sequences
        and (1162, 2594) in keys_to_sequences
        and (1662, 2603) in keys_to_sequences
    )


def test_detect_box_sequences_3():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/03.jpg")}
    assert (
        (1395, 343) in keys_to_sequences
        and (1460, 545) in keys_to_sequences
        and (1193, 2681) in keys_to_sequences
        and (1478, 2635) in keys_to_sequences  # b, x, b, b
        and (1686, 2603) in keys_to_sequences
    )


def test_detect_box_sequences_4():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/04.jpg")}
    assert (
        (2492, 610) in keys_to_sequences
        and (2457, 861) in keys_to_sequences
        and (1628, 1828) in keys_to_sequences
        and (2548, 1845) in keys_to_sequences
    )


def test_detect_box_sequences_5():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/05.jpg")}
    assert (
        (2676, 340) in keys_to_sequences
        and (2745, 663) in keys_to_sequences
        and (2549, 978) in keys_to_sequences
        and (577, 1869) in keys_to_sequences
        and (1685, 1864) in keys_to_sequences
        and (2826, 1871) in keys_to_sequences
    )


def test_detect_box_sequences_6():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/06.jpg")}
    assert (
        (1547, 579) in keys_to_sequences
        and (715, 2627) in keys_to_sequences
        and (1236, 2623) in keys_to_sequences
    )


def test_detect_box_sequences_7():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/07.jpg")}
    assert (
        (1524, 340) in keys_to_sequences
        and (1556, 566) in keys_to_sequences
        and (628, 2587) in keys_to_sequences
        and (1113, 2673) in keys_to_sequences
        and (1634, 2772) in keys_to_sequences
    )


def test_detect_box_sequences_8():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/08.jpg")}
    assert (
        (1601, 315) in keys_to_sequences
        and (1623, 551) in keys_to_sequences
        and (679, 2683) in keys_to_sequences
    )


def test_detect_box_sequences_9():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/09.jpg")}
    assert (
        (2371, 1087) in keys_to_sequences
        and (2540, 1581) in keys_to_sequences
    )


def test_detect_box_sequences_10():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/10.jpg")}
    assert (
        (1536, 286) in keys_to_sequences
        and (1605, 550) in keys_to_sequences
        and (613, 2761) in keys_to_sequences
        and (1157, 2755) in keys_to_sequences
        and (1693, 2757) in keys_to_sequences
    )


def test_detect_box_sequences_11():
    keys_to_sequences = {s.key: s for s in detect_box_sequences(HERE + "/11.jpg")}
    assert (
        (551, 616) in keys_to_sequences
        and (600, 743) in keys_to_sequences
    )
