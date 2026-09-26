# Variance matching test

import pandas as pd
from scripts.variance_matched_control import match_positions


def test_matching_does_not_use_model_results():
    effects = [
        0, 1, 2, 10, 20, 21, 30, 31,
        3, 11, 12, 13, 22, 23, 32, 33,
        4, 5, 14, 15, 24, 25, 34, 35,
        6, 7, 16, 17, 26, 27, 36, 37,
    ]
    position_table = pd.DataFrame({
        "DMS_id": ["A"] * 32,
        "position": list(range(1, 33)),
        "entropy": list(range(32)),
        "mean_abs_DMS_score": effects,
    })
    first_match = match_positions(position_table)

    position_table["esm2_score"] = list(range(32))
    position_table["correctness"] = 1
    position_table["controlled_rho"] = -1
    second_match = match_positions(position_table)

    assert first_match.equals(second_match)
    assert first_match["selected"].sum() == 24
