
import pytest

from agent.tools import (
    calculator,
    revenue_growth,
    net_profit_margin,
    debt_to_equity,
)


# ---------------- Calculator Tests ----------------

@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 3", "5"),
        ("10 - 4", "6"),
        ("5 * 4", "20"),
        ("20 / 4", "5.0"),
        ("(500 - 400) / 400 * 100", "25.0"),
        ("2 ** 3", "8"),
        ("-5 + 10", "5"),
    ]
)
def test_calculator_valid_expressions(expression, expected):
    result = calculator.invoke({"expression": expression})
    assert result == expected


def test_calculator_division_by_zero():
    result = calculator.invoke({"expression": "10 / 0"})
    assert result == "Error: Division by zero."


@pytest.mark.parametrize(
    "expression",
    [
        "import os",
        "open('test.txt')",
        "x + 5",
        "2 ** 11",
        "5 % 2",
    ]
)
def test_calculator_rejects_unsupported_expressions(expression):
    result = calculator.invoke({"expression": expression})
    assert result.startswith("Calculation error:")


# ---------------- Financial Tool Tests ----------------

def test_revenue_growth():
    result = revenue_growth.invoke({
        "previous_revenue": 100,
        "current_revenue": 120
    })
    assert result == "Revenue growth: 20.00%"


def test_revenue_growth_zero_previous_revenue():
    result = revenue_growth.invoke({
        "previous_revenue": 0,
        "current_revenue": 120
    })
    assert result == "Error: Previous revenue cannot be zero."


def test_net_profit_margin():
    result = net_profit_margin.invoke({
        "net_income": 25,
        "revenue": 100
    })
    assert result == "Net profit margin: 25.00%"


def test_net_profit_margin_zero_revenue():
    result = net_profit_margin.invoke({
        "net_income": 25,
        "revenue": 0
    })
    assert result == "Error: Revenue cannot be zero."


def test_debt_to_equity():
    result = debt_to_equity.invoke({
        "total_debt": 200,
        "total_equity": 100
    })
    assert result == "Debt-to-equity ratio: 2.00"


def test_debt_to_equity_zero_equity():
    result = debt_to_equity.invoke({
        "total_debt": 200,
        "total_equity": 0
    })
    assert result == "Error: Total equity cannot be zero."
