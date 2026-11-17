"""Tests for the event-to-label derivation in src.data.metrica_sample (synthetic events only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.common import AWAY, DEAD, HOME, UNLABELLED
from src.data.metrica_sample import derive_labels


def events(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start_frame", "end_frame"])


def test_kickoff_turnover_ball_out_and_restart():
    ev = events([
        ("H", "SET PIECE", "KICK OFF", 1, 3, 3),
        ("H", "PASS", "", 1, 3, 6),
        ("H", "BALL LOST", "INTERCEPTION", 1, 8, 9),   # A has it from frame 9
        ("A", "RECOVERY", "INTERCEPTION", 1, 10, 10),
        ("A", "BALL OUT", "", 1, 12, 13),               # dead from frame 13
        ("H", "SET PIECE", "THROW IN", 1, 16, 16),      # alive, H, from frame 16
        ("H", "PASS", "", 1, 16, 18),
    ])
    period = np.ones(20, np.int8)
    labels, diag = derive_labels(ev, 20, period)
    # row = frame - 1
    assert (labels[:2] == UNLABELLED).all()                     # before kick-off
    assert (labels[2:8] == HOME).all()                          # frames 3..8
    assert (labels[8:12] == AWAY).all()                         # frames 9..12
    assert (labels[12:15] == DEAD).all()                        # frames 13..15
    assert (labels[15:18] == HOME).all()                        # frames 16..18
    assert (labels[18:] == UNLABELLED).all()                    # after the last event
    assert diag["unlogged_restarts"] == 0


def test_keeper_recovery_keeps_dead_and_goal_kick_restarts():
    ev = events([
        ("H", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("H", "SHOT", "OFF TARGET-OUT", 1, 2, 4),       # dead from frame 4
        ("A", "RECOVERY", "", 1, 6, 6),                  # keeper picks it up: still dead
        ("A", "PASS", "GOAL KICK", 1, 9, 12),            # restart
    ])
    labels, diag = derive_labels(ev, 12, np.ones(12, np.int8))
    assert (labels[3:8] == DEAD).all()
    assert (labels[8:12] == AWAY).all()
    assert diag["unlogged_restarts"] == 0


def test_fault_is_dead_until_free_kick_and_unlogged_restart_is_counted():
    ev = events([
        ("H", "SET PIECE", "KICK OFF", 1, 1, 1),
        ("A", "FAULT RECEIVED", "", 1, 3, 3),            # dead from frame 3
        ("A", "SET PIECE", "FREE KICK", 1, 6, 6),
        ("A", "BALL OUT", "", 1, 8, 9),                  # dead from frame 9
        ("H", "PASS", "", 1, 11, 14),                    # no restart logged: alive again, counted
    ])
    labels, diag = derive_labels(ev, 14, np.ones(14, np.int8))
    assert (labels[2:5] == DEAD).all() and (labels[5:8] == AWAY).all()
    assert (labels[8:10] == DEAD).all() and (labels[10:14] == HOME).all()
    assert diag["unlogged_restarts"] == 1
