"""Tests for the numericvision.contours module."""
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
        and (1478, 2635) in keys_to_sequences # b, x, b, b
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
