"""
SUPERENGINE - AST Parser
========================
This module is responsible for parsing Excel formulas into an Abstract Syntax Tree (AST).
It uses the Lark parsing library to convert a formula string into a structured,
tree-like representation that the compiler can then walk and execute.

Key Innovation: Moving from regex/pattern-matching to a formal grammar-based parser.
This allows us to handle arbitrarily complex and nested formulas, providing the
foundation for a true formula compiler.
"""

import logging
from lark import Lark, Transformer, v_args, Tree

logger = logging.getLogger(__name__)

# --- LARK GRAMMAR FOR EXCEL FORMULAS ---
# Enhanced grammar that properly handles Excel syntax including ranges and functions
EXCEL_FORMULA_GRAMMAR = r"""
    ?start: "=" expression

    ?expression: or_expr

    ?or_expr: and_expr ("OR" and_expr)*
    ?and_expr: comparison ("AND" comparison)*
    
    ?comparison: addition (COMP_OP addition)?
    COMP_OP: ">=" | "<=" | "<>" | "!=" | ">" | "<" | "="

    ?addition: term (ADD_OP term)*
    ADD_OP: "+" | "-"
    
    ?term: factor (MUL_OP factor)*
    MUL_OP: "*" | "/"

    ?factor: power
    ?power: unary ("^" factor)?
    
    ?unary: "+" unary -> pos
          | "-" unary -> neg
          | atom

    ?atom: NUMBER           -> number
         | STRING           -> string
         | TRUE             -> true_const
         | FALSE            -> false_const
         | cell_range       -> range
         | cell             -> cell
         | table_ref        -> table_reference
         | function_call
         | identifier       -> named_range
         | "(" expression ")"

    cell: SHEET_REF? COL ROW
    cell_range: cell ":" cell
    table_ref: identifier "[" identifier "]"
    identifier: IDENTIFIER
    
    function_call: FUNC_NAME "(" [arguments] ")"
    arguments: expression ("," expression)*

    SHEET_REF: (IDENTIFIER | QUOTED_SHEET) "!"
    QUOTED_SHEET: "'" /[^']+/ "'"
    
    COL: /\$?[A-Z]+/
    ROW: /\$?\d+/
    
    FUNC_NAME: /[A-Z][A-Z0-9_]*/
    IDENTIFIER: /[A-Za-z_][A-Za-z0-9_]*/
    
    STRING: /"[^"]*"/
    NUMBER: /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/
    TRUE: /TRUE/i
    FALSE: /FALSE/i

    %import common.WS
    %ignore WS
"""

class AstTransformer(Transformer):
    """
    Transforms the Lark parse tree into a more usable AST node structure.
    Each method corresponds to a rule in the grammar.
    """
    
    def start(self, args):
        return args[0]
    
    def expression(self, args):
        return args[0]
    
    def or_expr(self, args):
        if len(args) == 1:
            return args[0]
        # Build OR tree
        result = args[0]
        for i in range(1, len(args)):
            result = ('function_call', 'OR', [result, args[i]])
        return result
    
    def and_expr(self, args):
        if len(args) == 1:
            return args[0]
        # Build AND tree
        result = args[0]
        for i in range(1, len(args)):
            result = ('function_call', 'AND', [result, args[i]])
        return result
    
    def comparison(self, args):
        if len(args) == 1:
            return args[0]
        left, op, right = args
        op_map = {
            '>': 'gt', '<': 'lt', '>=': 'gte', '<=': 'lte', 
            '=': 'eq', '<>': 'neq', '!=': 'neq'
        }
        return (op_map[str(op)], left, right)
    
    def addition(self, args):
        if len(args) == 1:
            return args[0]
        result = args[0]
        for i in range(1, len(args), 2):
            op = str(args[i])
            right = args[i + 1]
            if op == '+':
                result = ('add', result, right)
            else:  # '-'
                result = ('sub', result, right)
        return result
    
    def term(self, args):
        if len(args) == 1:
            return args[0]
        result = args[0]
        for i in range(1, len(args), 2):
            op = str(args[i])
            right = args[i + 1]
            if op == '*':
                result = ('mul', result, right)
            else:  # '/'
                result = ('div', result, right)
        return result
    
    def factor(self, args):
        return args[0]
    
    def power(self, args):
        if len(args) == 1:
            return args[0]
        return ('power', args[0], args[1])
    
    def neg(self, args):
        return ('neg', args[0])
    
    def pos(self, args):
        return args[0]
    
    def number(self, args):
        return ('number', float(args[0]))
    
    def string(self, args):
        # Remove quotes
        s = str(args[0])
        return ('string', s[1:-1])
    
    def true_const(self, args):
        return ('bool', True)
    
    def false_const(self, args):
        return ('bool', False)
    
    def cell(self, args):
        if len(args) == 3:  # Sheet reference exists
            sheet = str(args[0]).rstrip('!')
            if sheet.startswith("'") and sheet.endswith("'"):
                sheet = sheet[1:-1]
            col = str(args[1])
            row = str(args[2])
            return ('cell', f"{sheet}!{col}{row}")
        elif len(args) == 2:  # No sheet reference
            col = str(args[0])
            row = str(args[1])
            return ('cell', f"{col}{row}")
        else:
            # Fallback
            return ('cell', ''.join(str(arg) for arg in args))
    
    def range(self, args):
        # args[0] is the cell_range Tree
        if hasattr(args[0], 'children'):
            children = args[0].children
            if len(children) >= 2:
                start_cell = children[0]
                end_cell = children[1]
                # Process cells
                start = self.cell(start_cell.children if hasattr(start_cell, 'children') else [start_cell])
                end = self.cell(end_cell.children if hasattr(end_cell, 'children') else [end_cell])
                return ('range', start[1], end[1])
        return ('range', 'A1', 'A1')  # Fallback
    
    def named_range(self, args):
        return ('named_range', str(args[0]))
    
    def table_reference(self, args):
        if hasattr(args[0], 'children'):
            children = args[0].children
            if len(children) >= 2:
                return ('table_ref', str(children[0]), str(children[1]))
        return ('table_ref', '', '')  # Fallback
    
    def function_call(self, args):
        if hasattr(args[0], 'children'):
            children = args[0].children
            func_name = str(children[0]).upper()
            if len(children) > 1 and hasattr(children[1], 'children'):
                # Has arguments
                func_args = children[1].children
                return ('function_call', func_name, func_args)
            else:
                return ('function_call', func_name, [])
        return ('function_call', 'UNKNOWN', [])
    
    def arguments(self, args):
        return list(args)
    
    def identifier(self, args):
        return str(args[0])
    
    # Token transformers
    def COMP_OP(self, token):
        return str(token)
    
    def ADD_OP(self, token):
        return str(token)
    
    def MUL_OP(self, token):
        return str(token)
    
    def NUMBER(self, token):
        return str(token)
    
    def STRING(self, token):
        return str(token)
    
    def IDENTIFIER(self, token):
        return str(token)
    
    def FUNC_NAME(self, token):
        return str(token)
    
    def COL(self, token):
        return str(token)
    
    def ROW(self, token):
        return str(token)
    
    def SHEET_REF(self, token):
        return str(token)


class FormulaParser:
    """
    Parses an Excel formula string into an Abstract Syntax Tree (AST).
    """
    def __init__(self):
        try:
            # Don't apply transformer during parsing, apply it after
            self.parser = Lark(EXCEL_FORMULA_GRAMMAR, start='start', parser='lalr')
            self.transformer = AstTransformer()
            logger.info("✅ SUPERENGINE: FormulaParser initialized with LALR parser.")
        except Exception as e:
            logger.error(f"❌ SUPERENGINE: Failed to initialize Lark parser: {e}", exc_info=True)
            raise

    def parse(self, formula_string: str) -> tuple:
        """
        Parses a given formula string and returns its AST representation.

        Args:
            formula_string: The Excel formula to parse (e.g., "=SUM(A1:B2, C3) * 2").

        Returns:
            A tuple-based AST representing the formula.
            Example: ('mul', ('function_call', 'SUM', [('range', 'A1', 'B2'), ('cell', 'C3')]), ('number', 2.0))
        """
        if not isinstance(formula_string, str) or not formula_string.startswith('='):
            raise ValueError("Invalid formula string. Must start with '='.")

        try:
            # Parse to tree first
            tree = self.parser.parse(formula_string)
            # Then transform
            ast = self.transformer.transform(tree)
            logger.debug(f"Successfully parsed '{formula_string}' to AST: {ast}")
            return ast
        except Exception as e:
            logger.error(f"Failed to parse formula '{formula_string}': {e}", exc_info=True)
            raise ValueError(f"Could not parse formula: {formula_string}")


# Example usage for testing and demonstration
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = FormulaParser()

    test_formulas = [
        "=A1+B1",
        "=A1>B1", 
        "=SUM(A1:A10)",
        "=IF(A1>100, B1*2, B1/2)",
        "=VLOOKUP(A1, D1:E10, 2, FALSE)",
        "=A1+B1*C1-D1/E1",
        "=(A1+B1)*(C1-D1)",
        "=MIN(A1:A10)",
        "=MAX(B1:B10)",
        "=AVERAGE(C1:C10)",
        "=AND(A1>0, B1<100)",
        "=OR(A1=0, B1=100)",
        "=NOT(A1>50)",
        '=SUMIF(A1:A10, ">50", B1:B10)',
        "=MyNamedRange + A1",
        "=Sales[Revenue] * 0.1",
        "=100*(A2+B2)/SUM(C1:C10)"
    ]

    for f in test_formulas:
        try:
            ast_representation = parser.parse(f)
            print(f"Formula: {f}\nAST: {ast_representation}\n")
        except Exception as e:
            print(f"Formula: {f}\nError: {e}\n")

    # Example of a more complex AST
    complex_formula = "=SUM(A1, B1*2) + AVERAGE(C1:C10)"
    try:
        ast = parser.parse(complex_formula)
        print(f"Complex Formula: {complex_formula}")
        import json
        print(f"Pretty AST: {json.dumps(ast, indent=2)}")
    except Exception as e:
        print(f"Complex Formula: {complex_formula}\nError: {e}\n")
