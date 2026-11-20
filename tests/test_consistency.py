import pytest

from obesity.consistency import bmi, bmi_to_class


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (15.0, "Insufficient_Weight"),
        (18.49, "Insufficient_Weight"),
        (18.5, "Normal_Weight"),
        (24.99, "Normal_Weight"),
        (25.0, "Overweight_Level_I"),
        (26.89, "Overweight_Level_I"),
        (26.9, "Overweight_Level_II"),
        (29.99, "Overweight_Level_II"),
        (30.0, "Obesity_Type_I"),
        (35.0, "Obesity_Type_II"),
        (40.0, "Obesity_Type_III"),
        (55.0, "Obesity_Type_III"),
    ],
)
def test_bmi_boundaries(value, expected):
    assert bmi_to_class([value])[0] == expected


def test_bmi_formula():
    assert bmi(80, 2.0) == 20.0
