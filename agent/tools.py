import ast
import operator

from langchain_core.tools import tool


# Only allow these arithmetic operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        left = safe_evaluate(node.left)
        right = safe_evaluate(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent is too large.")

        return ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](
            safe_evaluate(node.operand)
        )

    raise ValueError("Unsupported mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Example: (500 - 400) / 400 * 100."""

    try:
        tree = ast.parse(expression, mode="eval")
        result = safe_evaluate(tree.body)

        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero."

    except Exception as e:
        return f"Calculation error: {e}"



@tool
def revenue_growth(previous_revenue: float, current_revenue: float) -> str:
    """Calculate revenue growth percentage using previous and current revenue."""

    if previous_revenue == 0:
        return "Error: Previous revenue cannot be zero."

    growth = (
        (current_revenue - previous_revenue)
        / previous_revenue
    ) * 100

    return f"Revenue growth: {growth:.2f}%"


@tool
def net_profit_margin(net_income: float, revenue: float) -> str:
    """Calculate net profit margin percentage using net income and revenue."""

    if revenue == 0:
        return "Error: Revenue cannot be zero."

    margin = (net_income / revenue) * 100

    return f"Net profit margin: {margin:.2f}%"


@tool
def debt_to_equity(total_debt: float, total_equity: float) -> str:
    """Calculate the debt-to-equity ratio using total debt and total equity."""

    if total_equity == 0:
        return "Error: Total equity cannot be zero."

    ratio = total_debt / total_equity

    return f"Debt-to-equity ratio: {ratio:.2f}"