"""
SUPERENGINE - Hybrid Excel Formula Parser
=========================================
A world-class parser that combines the best of regex preprocessing with
AST-based parsing to achieve perfect Excel formula parsing.

This parser uses a two-phase approach:
1. Tokenization phase: Uses regex to identify and tag Excel constructs
2. AST phase: Parses the tagged tokens into a proper AST

This approach solves the ambiguity between cell references and function names.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum, auto
import json

logger = logging.getLogger(__name__)

# Token Types
class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    ERROR = auto()
    
    # References
    CELL = auto()
    RANGE = auto()
    NAMED_RANGE = auto()
    TABLE_REF = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    CONCAT = auto()
    
    # Comparison
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Structural
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    
    # Functions
    FUNCTION = auto()
    
    # Special
    SHEET_PREFIX = auto()
    ABSOLUTE = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    position: int

# AST Node Types (reuse from parser_v2)
@dataclass
class ASTNode:
    """Base class for all AST nodes"""
    pass

@dataclass
class NumberNode(ASTNode):
    value: float

@dataclass
class StringNode(ASTNode):
    value: str

@dataclass
class BooleanNode(ASTNode):
    value: bool

@dataclass
class ErrorNode(ASTNode):
    error_type: str

@dataclass
class CellNode(ASTNode):
    sheet: Optional[str]
    column: str
    row: int
    absolute_col: bool = False
    absolute_row: bool = False

@dataclass
class RangeNode(ASTNode):
    start: CellNode
    end: CellNode

@dataclass
class NamedRangeNode(ASTNode):
    name: str
    scope: Optional[str] = None

@dataclass
class TableRefNode(ASTNode):
    table_name: str
    column_spec: Optional[str] = None
    special_item: Optional[str] = None

@dataclass
class FunctionNode(ASTNode):
    name: str
    args: List[ASTNode]

@dataclass
class BinaryOpNode(ASTNode):
    op: TokenType
    left: ASTNode
    right: ASTNode

@dataclass
class UnaryOpNode(ASTNode):
    op: TokenType
    operand: ASTNode

@dataclass
class ArrayNode(ASTNode):
    elements: List[List[ASTNode]]

class ExcelTokenizer:
    """
    Phase 1: Tokenize Excel formulas using regex patterns.
    """
    
    # Excel function names (comprehensive list)
    EXCEL_FUNCTIONS = {
        # Math & Trig
        'ABS', 'ACOS', 'ACOSH', 'ASIN', 'ASINH', 'ATAN', 'ATAN2', 'ATANH',
        'CEILING', 'COS', 'COSH', 'EXP', 'FLOOR', 'INT', 'LN', 'LOG', 'LOG10',
        'MOD', 'PI', 'POWER', 'ROUND', 'ROUNDDOWN', 'ROUNDUP', 'SIGN', 'SIN',
        'SINH', 'SQRT', 'TAN', 'TANH', 'TRUNC',
        
        # Statistical
        'AVERAGE', 'AVERAGEA', 'AVERAGEIF', 'AVERAGEIFS', 'COUNT', 'COUNTA',
        'COUNTBLANK', 'COUNTIF', 'COUNTIFS', 'MAX', 'MAXA', 'MIN', 'MINA',
        'MEDIAN', 'MODE', 'PERCENTILE', 'QUARTILE', 'STDEV', 'STDEVA',
        'STDEVP', 'STDEVPA', 'VAR', 'VARA', 'VARP', 'VARPA',
        
        # Lookup & Reference
        'ADDRESS', 'CHOOSE', 'COLUMN', 'COLUMNS', 'HLOOKUP', 'INDEX',
        'INDIRECT', 'LOOKUP', 'MATCH', 'OFFSET', 'ROW', 'ROWS', 'VLOOKUP',
        
        # Text
        'CHAR', 'CLEAN', 'CODE', 'CONCATENATE', 'EXACT', 'FIND', 'LEFT',
        'LEN', 'LOWER', 'MID', 'PROPER', 'REPLACE', 'REPT', 'RIGHT',
        'SEARCH', 'SUBSTITUTE', 'TEXT', 'TRIM', 'UPPER', 'VALUE',
        
        # Logical
        'AND', 'FALSE', 'IF', 'IFERROR', 'IFNA', 'IFS', 'NOT', 'OR',
        'SWITCH', 'TRUE', 'XOR',
        
        # Date & Time
        'DATE', 'DATEVALUE', 'DAY', 'DAYS', 'EDATE', 'EOMONTH', 'HOUR',
        'MINUTE', 'MONTH', 'NOW', 'SECOND', 'TIME', 'TIMEVALUE', 'TODAY',
        'WEEKDAY', 'WEEKNUM', 'WORKDAY', 'YEAR', 'YEARFRAC',
        
        # Financial
        'FV', 'IPMT', 'IRR', 'MIRR', 'NPER', 'NPV', 'PMT', 'PPMT', 'PV',
        'RATE', 'SLN', 'SYD', 'VDB', 'XIRR', 'XNPV',
        
        # Information
        'CELL', 'ERROR.TYPE', 'INFO', 'ISBLANK', 'ISERR', 'ISERROR',
        'ISEVEN', 'ISLOGICAL', 'ISNA', 'ISNONTEXT', 'ISNUMBER', 'ISODD',
        'ISREF', 'ISTEXT', 'N', 'NA', 'TYPE',
        
        # Database
        'DAVERAGE', 'DCOUNT', 'DCOUNTA', 'DGET', 'DMAX', 'DMIN', 'DPRODUCT',
        'DSTDEV', 'DSTDEVP', 'DSUM', 'DVAR', 'DVARP',
        
        # Engineering
        'CONVERT', 'DEC2BIN', 'DEC2HEX', 'DEC2OCT', 'DELTA', 'ERF', 'ERFC',
        'GESTEP', 'HEX2BIN', 'HEX2DEC', 'HEX2OCT', 'IMABS', 'IMAGINARY',
        'IMARGUMENT', 'IMCONJUGATE', 'IMCOS', 'IMDIV', 'IMEXP', 'IMLN',
        'IMLOG10', 'IMLOG2', 'IMPOWER', 'IMPRODUCT', 'IMREAL', 'IMSIN',
        'IMSQRT', 'IMSUB', 'IMSUM', 'OCT2BIN', 'OCT2DEC', 'OCT2HEX',
        
        # Array formulas
        'TRANSPOSE', 'MMULT', 'MINVERSE', 'MDETERM',
        
        # Custom/Monte Carlo
        'NORMAL', 'LOGNORMAL', 'TRIANGULAR', 'UNIFORM', 'BETA', 'GAMMA',
        'EXPONENTIAL', 'POISSON', 'BINOMIAL',
    }
    
    # Regex patterns
    PATTERNS = {
        # Sheet reference (e.g., Sheet1! or 'My Sheet'!)
        'SHEET_PREFIX': r"(?:'([^']+)'|([A-Za-z_]\w*))!",
        
        # Cell reference with optional absolute markers
        'CELL': r'(\$?)([A-Z]{1,3})(\$?)(\d{1,7})\b',
        
        # Range (handled separately after cells are identified)
        
        # Numbers (including scientific notation)
        'NUMBER': r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?',
        
        # Strings
        'STRING': r'"(?:[^"\\]|\\.)*"',
        
        # Boolean
        'BOOLEAN': r'\b(?:TRUE|FALSE)\b',
        
        # Error values
        'ERROR': r'#(?:NULL!|DIV/0!|VALUE!|REF!|NAME\?|NUM!|N/A|SPILL!)',
        
        # Table reference
        'TABLE_REF': r'([A-Za-z_]\w*)\[((?:#(?:All|Data|Headers|Totals|This\s+Row))|(?:\[[^\]]+\])|(?:[^\[\]]+))\]',
        
        # Operators and delimiters
        'OPERATORS': r'[+\-*/^&<>=]|<>|<=|>=',
        'DELIMITERS': r'[(),;:{}[\]]',
        
        # Whitespace
        'WHITESPACE': r'\s+',
    }
    
    def __init__(self):
        # Compile regex patterns
        self.sheet_pattern = re.compile(self.PATTERNS['SHEET_PREFIX'])
        self.cell_pattern = re.compile(self.PATTERNS['CELL'])
        self.number_pattern = re.compile(self.PATTERNS['NUMBER'])
        self.string_pattern = re.compile(self.PATTERNS['STRING'])
        self.boolean_pattern = re.compile(self.PATTERNS['BOOLEAN'], re.IGNORECASE)
        self.error_pattern = re.compile(self.PATTERNS['ERROR'])
        self.table_pattern = re.compile(self.PATTERNS['TABLE_REF'])
        self.operator_pattern = re.compile(self.PATTERNS['OPERATORS'])
        self.delimiter_pattern = re.compile(self.PATTERNS['DELIMITERS'])
        self.whitespace_pattern = re.compile(self.PATTERNS['WHITESPACE'])
        
        # Function pattern (case-insensitive)
        func_names = '|'.join(self.EXCEL_FUNCTIONS)
        self.function_pattern = re.compile(f'\\b({func_names})\\b(?=\\s*\\()', re.IGNORECASE)
        
        # Named range pattern (anything that's not a cell or function)
        self.name_pattern = re.compile(r'[A-Za-z_]\w*')
    
    def tokenize(self, formula: str) -> List[Token]:
        """
        Tokenize an Excel formula.
        
        Args:
            formula: Excel formula (with or without =)
            
        Returns:
            List of tokens
        """
        # Remove leading = if present
        if formula.startswith('='):
            formula = formula[1:]
        
        tokens = []
        position = 0
        
        while position < len(formula):
            # Skip whitespace
            match = self.whitespace_pattern.match(formula, position)
            if match:
                position = match.end()
                continue
            
            # Try each pattern in order
            matched = False
            
            # 1. Error values (highest priority)
            match = self.error_pattern.match(formula, position)
            if match:
                tokens.append(Token(TokenType.ERROR, match.group(), position))
                position = match.end()
                matched = True
                continue
            
            # 2. Strings
            match = self.string_pattern.match(formula, position)
            if match:
                tokens.append(Token(TokenType.STRING, match.group(), position))
                position = match.end()
                matched = True
                continue
            
            # 3. Numbers
            match = self.number_pattern.match(formula, position)
            if match:
                # Check if it's actually part of a cell reference
                if position > 0 and formula[position-1].isalpha():
                    # This is part of a cell reference, not a standalone number
                    pass
                else:
                    # Check if this is a signed number that's part of an expression
                    num_str = match.group()
                    if num_str[0] in '+-' and position > 0:
                        # This might be an operator followed by a number
                        # Only treat as a single number if preceded by certain tokens
                        prev_char = formula[position-1]
                        if prev_char in '(),=':
                            # It's a signed number
                            tokens.append(Token(TokenType.NUMBER, num_str, position))
                            position = match.end()
                            matched = True
                            continue
                        else:
                            # It's an operator followed by unsigned number
                            # Don't match the sign
                            unsigned_match = re.match(r'\d+\.?\d*(?:[eE][+-]?\d+)?', formula[position+1:])
                            if unsigned_match:
                                # First add the operator
                                op = num_str[0]
                                token_type = TokenType.PLUS if op == '+' else TokenType.MINUS
                                tokens.append(Token(token_type, op, position))
                                position += 1
                                # Then add the number
                                tokens.append(Token(TokenType.NUMBER, unsigned_match.group(), position))
                                position += len(unsigned_match.group())
                                matched = True
                                continue
                    else:
                        tokens.append(Token(TokenType.NUMBER, num_str, position))
                        position = match.end()
                        matched = True
                        continue
            
            # 4. Booleans
            match = self.boolean_pattern.match(formula, position)
            if match:
                tokens.append(Token(TokenType.BOOLEAN, match.group().upper(), position))
                position = match.end()
                matched = True
                continue
            
            # 5. Table references
            match = self.table_pattern.match(formula, position)
            if match:
                tokens.append(Token(TokenType.TABLE_REF, match.group(), position))
                position = match.end()
                matched = True
                continue
            
            # 6. Sheet prefix + cell reference
            match = self.sheet_pattern.match(formula, position)
            if match:
                sheet_name = match.group(1) or match.group(2)
                tokens.append(Token(TokenType.SHEET_PREFIX, sheet_name, position))
                position = match.end()
                
                # Check if followed by cell reference
                cell_match = self.cell_pattern.match(formula, position)
                if cell_match:
                    tokens.append(Token(TokenType.CELL, cell_match.group(), position))
                    position = cell_match.end()
                    matched = True
                    continue
            
            # 7. Cell references (without sheet prefix)
            match = self.cell_pattern.match(formula, position)
            if match:
                tokens.append(Token(TokenType.CELL, match.group(), position))
                position = match.end()
                matched = True
                continue
            
            # 8. Functions (before general names)
            match = self.function_pattern.match(formula, position)
            if match:
                tokens.append(Token(TokenType.FUNCTION, match.group().upper(), position))
                position = match.end()
                matched = True
                continue
            
            # 8b. Check for function-like names followed by (
            # This catches functions that aren't in our list yet
            if position < len(formula):
                name_match = self.name_pattern.match(formula, position)
                if name_match:
                    name = name_match.group()
                    next_pos = name_match.end()
                    while next_pos < len(formula) and formula[next_pos].isspace():
                        next_pos += 1
                    if next_pos < len(formula) and formula[next_pos] == '(':
                        # It's a function!
                        tokens.append(Token(TokenType.FUNCTION, name.upper(), position))
                        position = name_match.end()
                        matched = True
                        continue
            
            # 9. Operators
            if position < len(formula) and formula[position:position+2] in ['<>', '<=', '>=']:
                op = formula[position:position+2]
                token_type = {
                    '<>': TokenType.NEQ,
                    '<=': TokenType.LTE,
                    '>=': TokenType.GTE
                }[op]
                tokens.append(Token(token_type, op, position))
                position += 2
                matched = True
                continue
            
            if position < len(formula) and formula[position] in '+-*/^&<>=':
                op = formula[position]
                token_type = {
                    '+': TokenType.PLUS,
                    '-': TokenType.MINUS,
                    '*': TokenType.MULTIPLY,
                    '/': TokenType.DIVIDE,
                    '^': TokenType.POWER,
                    '&': TokenType.CONCAT,
                    '<': TokenType.LT,
                    '>': TokenType.GT,
                    '=': TokenType.EQ,
                }[op]
                tokens.append(Token(token_type, op, position))
                position += 1
                matched = True
                continue
            
            # 10. Delimiters
            if position < len(formula) and formula[position] in '(),;:{}[]':
                delim = formula[position]
                token_type = {
                    '(': TokenType.LPAREN,
                    ')': TokenType.RPAREN,
                    ',': TokenType.COMMA,
                    ';': TokenType.SEMICOLON,
                    ':': TokenType.COLON,
                    '{': TokenType.LBRACE,
                    '}': TokenType.RBRACE,
                    '[': TokenType.LBRACKET,
                    ']': TokenType.RBRACKET,
                }[delim]
                tokens.append(Token(token_type, delim, position))
                position += 1
                matched = True
                continue
            
            # 11. Named ranges or other identifiers
            match = self.name_pattern.match(formula, position)
            if match:
                name = match.group()
                # Check if it's AND/OR/NOT
                if name.upper() in ['AND', 'OR', 'NOT']:
                    # Check if followed by (
                    next_pos = match.end()
                    while next_pos < len(formula) and formula[next_pos].isspace():
                        next_pos += 1
                    if next_pos < len(formula) and formula[next_pos] == '(':
                        tokens.append(Token(TokenType.FUNCTION, name.upper(), position))
                    else:
                        token_type = {
                            'AND': TokenType.AND,
                            'OR': TokenType.OR,
                            'NOT': TokenType.NOT,
                        }[name.upper()]
                        tokens.append(Token(token_type, name.upper(), position))
                else:
                    tokens.append(Token(TokenType.NAMED_RANGE, name, position))
                position = match.end()
                matched = True
                continue
            
            # If nothing matched, skip character
            if not matched:
                logger.warning(f"Unrecognized character at position {position}: {formula[position]}")
                position += 1
        
        return tokens

class ExcelParser:
    """
    Phase 2: Parse tokens into AST.
    """
    
    def __init__(self, tokens: List[Token], tokenizer: ExcelTokenizer = None):
        self.tokens = tokens
        self.position = 0
        self.current_sheet = None
        self.tokenizer = tokenizer or ExcelTokenizer()
    
    def parse(self) -> ASTNode:
        """Parse tokens into AST."""
        if not self.tokens:
            raise ValueError("No tokens to parse")
        
        return self.parse_expression()
    
    def current_token(self) -> Optional[Token]:
        """Get current token without consuming it."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None
    
    def consume_token(self) -> Optional[Token]:
        """Consume and return current token."""
        if self.position < len(self.tokens):
            token = self.tokens[self.position]
            self.position += 1
            return token
        return None
    
    def peek_token(self, offset: int = 1) -> Optional[Token]:
        """Peek at token at offset from current position."""
        pos = self.position + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return None
    
    def parse_expression(self) -> ASTNode:
        """Parse expression with operator precedence."""
        return self.parse_or()
    
    def parse_or(self) -> ASTNode:
        """Parse OR expressions."""
        left = self.parse_and()
        
        while self.current_token() and self.current_token().type == TokenType.OR:
            self.consume_token()  # consume OR
            right = self.parse_and()
            left = BinaryOpNode(TokenType.OR, left, right)
        
        return left
    
    def parse_and(self) -> ASTNode:
        """Parse AND expressions."""
        left = self.parse_comparison()
        
        while self.current_token() and self.current_token().type == TokenType.AND:
            self.consume_token()  # consume AND
            right = self.parse_comparison()
            left = BinaryOpNode(TokenType.AND, left, right)
        
        return left
    
    def parse_comparison(self) -> ASTNode:
        """Parse comparison expressions."""
        left = self.parse_concatenation()
        
        token = self.current_token()
        if token and token.type in [TokenType.EQ, TokenType.NEQ, TokenType.LT,
                                     TokenType.GT, TokenType.LTE, TokenType.GTE]:
            op = self.consume_token().type
            right = self.parse_concatenation()
            return BinaryOpNode(op, left, right)
        
        return left
    
    def parse_concatenation(self) -> ASTNode:
        """Parse string concatenation."""
        left = self.parse_addition()
        
        while self.current_token() and self.current_token().type == TokenType.CONCAT:
            self.consume_token()  # consume &
            right = self.parse_addition()
            left = BinaryOpNode(TokenType.CONCAT, left, right)
        
        return left
    
    def parse_addition(self) -> ASTNode:
        """Parse addition and subtraction."""
        left = self.parse_multiplication()
        
        while self.current_token() and self.current_token().type in [TokenType.PLUS, TokenType.MINUS]:
            op = self.consume_token().type
            right = self.parse_multiplication()
            left = BinaryOpNode(op, left, right)
        
        return left
    
    def parse_multiplication(self) -> ASTNode:
        """Parse multiplication and division."""
        left = self.parse_power()
        
        while self.current_token() and self.current_token().type in [TokenType.MULTIPLY, TokenType.DIVIDE]:
            op = self.consume_token().type
            right = self.parse_power()
            left = BinaryOpNode(op, left, right)
        
        return left
    
    def parse_power(self) -> ASTNode:
        """Parse exponentiation."""
        left = self.parse_unary()
        
        if self.current_token() and self.current_token().type == TokenType.POWER:
            self.consume_token()  # consume ^
            right = self.parse_power()  # right associative
            return BinaryOpNode(TokenType.POWER, left, right)
        
        return left
    
    def parse_unary(self) -> ASTNode:
        """Parse unary expressions."""
        token = self.current_token()
        
        if token and token.type == TokenType.PLUS:
            self.consume_token()
            return self.parse_unary()  # +x is just x
        
        if token and token.type == TokenType.MINUS:
            self.consume_token()
            operand = self.parse_unary()
            return UnaryOpNode(TokenType.MINUS, operand)
        
        if token and token.type == TokenType.NOT:
            self.consume_token()
            operand = self.parse_unary()
            return UnaryOpNode(TokenType.NOT, operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        """Parse primary expressions."""
        token = self.current_token()
        
        if not token:
            raise ValueError("Unexpected end of expression")
        
        # Numbers
        if token.type == TokenType.NUMBER:
            self.consume_token()
            return NumberNode(float(token.value))
        
        # Strings
        if token.type == TokenType.STRING:
            self.consume_token()
            # Remove quotes and unescape
            value = token.value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
            return StringNode(value)
        
        # Booleans
        if token.type == TokenType.BOOLEAN:
            self.consume_token()
            return BooleanNode(token.value == 'TRUE')
        
        # Errors
        if token.type == TokenType.ERROR:
            self.consume_token()
            return ErrorNode(token.value)
        
        # Sheet prefix
        if token.type == TokenType.SHEET_PREFIX:
            sheet = self.consume_token().value
            # Next token should be a cell or column reference
            next_token = self.current_token()
            if next_token and next_token.type == TokenType.CELL:
                cell_token = self.consume_token()
                cell = self.parse_cell_token(cell_token.value)
                cell.sheet = sheet
                
                # Check for range
                if self.current_token() and self.current_token().type == TokenType.COLON:
                    self.consume_token()  # consume :
                    end_cell = self.parse_cell_or_reference()
                    if isinstance(end_cell, CellNode):
                        end_cell.sheet = sheet
                    return RangeNode(cell, end_cell)
                
                return cell
            elif next_token and next_token.type == TokenType.NAMED_RANGE:
                # Could be a column reference like A:D
                col_name = next_token.value
                if len(col_name) <= 3 and col_name.isalpha():
                    # It's likely a column reference
                    self.consume_token()
                    if self.current_token() and self.current_token().type == TokenType.COLON:
                        self.consume_token()  # consume :
                        end_token = self.current_token()
                        if end_token and end_token.type == TokenType.NAMED_RANGE and len(end_token.value) <= 3 and end_token.value.isalpha():
                            # Full column range like A:D
                            self.consume_token()
                            # Create cells representing full columns
                            start_cell = CellNode(sheet=sheet, column=col_name, row=1, absolute_col=False, absolute_row=False)
                            end_cell = CellNode(sheet=sheet, column=end_token.value, row=1048576, absolute_col=False, absolute_row=False)
                            return RangeNode(start_cell, end_cell)
                
                # Not a column reference, treat as named range
                return NamedRangeNode(col_name, scope=sheet)
            else:
                # Sheet prefix without cell - return as named range
                return NamedRangeNode(sheet)
        
        # Cells
        if token.type == TokenType.CELL:
            cell_token = self.consume_token()
            cell = self.parse_cell_token(cell_token.value)
            
            # Check for range
            if self.current_token() and self.current_token().type == TokenType.COLON:
                self.consume_token()  # consume :
                end_cell = self.parse_cell_or_reference()
                return RangeNode(cell, end_cell)
            
            return cell
        
        # Functions
        if token.type == TokenType.FUNCTION:
            return self.parse_function()
        
        # Table references
        if token.type == TokenType.TABLE_REF:
            self.consume_token()
            # Parse table reference
            match = re.match(r'([A-Za-z_]\w*)\[(.*)\]', token.value)
            if match:
                table_name = match.group(1)
                column_spec = match.group(2)
                
                # Check for special items
                if column_spec.startswith('#'):
                    return TableRefNode(table_name, special_item=column_spec)
                else:
                    # Remove brackets if present
                    if column_spec.startswith('[') and column_spec.endswith(']'):
                        column_spec = column_spec[1:-1]
                    return TableRefNode(table_name, column_spec=column_spec)
            
            raise ValueError(f"Invalid table reference: {token.value}")
        
        # Named ranges
        if token.type == TokenType.NAMED_RANGE:
            # Check if this is actually a function without parentheses
            # This happens when tokenizer misidentifies a function
            if token.value.upper() in self.tokenizer.EXCEL_FUNCTIONS and self.current_token() and self.current_token().type == TokenType.LPAREN:
                # It's actually a function!
                token = Token(TokenType.FUNCTION, token.value.upper(), token.position)
                return self.parse_function()
            
            self.consume_token()
            
            # Check for range
            if self.current_token() and self.current_token().type == TokenType.COLON:
                self.consume_token()  # consume :
                end = self.parse_cell_or_reference()
                if isinstance(end, NamedRangeNode):
                    # Named range : named range (not common but valid)
                    # Return as two separate nodes for now
                    return NamedRangeNode(token.value)
                else:
                    # Named range : cell (also uncommon)
                    return NamedRangeNode(token.value)
            
            return NamedRangeNode(token.value)
        
        # Parentheses
        if token.type == TokenType.LPAREN:
            self.consume_token()  # consume (
            expr = self.parse_expression()
            if self.current_token() and self.current_token().type == TokenType.RPAREN:
                self.consume_token()  # consume )
            else:
                raise ValueError("Expected closing parenthesis")
            return expr
        
        # Array constants
        if token.type == TokenType.LBRACE:
            return self.parse_array()
        
        raise ValueError(f"Unexpected token: {token.type} '{token.value}'")
    
    def parse_cell_token(self, cell_str: str) -> CellNode:
        """Parse a cell token string into a CellNode."""
        match = re.match(r'(\$?)([A-Z]+)(\$?)(\d+)', cell_str)
        if not match:
            raise ValueError(f"Invalid cell reference: {cell_str}")
        
        abs_col = match.group(1) == '$'
        column = match.group(2)
        abs_row = match.group(3) == '$'
        row = int(match.group(4))
        
        return CellNode(
            sheet=None,
            column=column,
            row=row,
            absolute_col=abs_col,
            absolute_row=abs_row
        )
    
    def parse_cell_or_reference(self) -> ASTNode:
        """Parse a cell or named range reference."""
        token = self.current_token()
        
        if token and token.type == TokenType.CELL:
            self.consume_token()
            return self.parse_cell_token(token.value)
        elif token and token.type == TokenType.NAMED_RANGE:
            self.consume_token()
            return NamedRangeNode(token.value)
        else:
            raise ValueError(f"Expected cell or named range, got {token}")
    
    def parse_function(self) -> FunctionNode:
        """Parse function call."""
        func_token = self.consume_token()
        func_name = func_token.value
        
        # Expect opening parenthesis
        if not self.current_token() or self.current_token().type != TokenType.LPAREN:
            raise ValueError(f"Expected '(' after function {func_name}")
        
        self.consume_token()  # consume (
        
        # Parse arguments
        args = []
        
        # Check for empty argument list
        if self.current_token() and self.current_token().type == TokenType.RPAREN:
            self.consume_token()  # consume )
            return FunctionNode(func_name, args)
        
        # Parse arguments
        while True:
            args.append(self.parse_expression())
            
            token = self.current_token()
            if token and token.type == TokenType.COMMA:
                self.consume_token()  # consume ,
                continue
            elif token and token.type == TokenType.RPAREN:
                self.consume_token()  # consume )
                break
            else:
                raise ValueError(f"Expected ',' or ')' in function arguments")
        
        return FunctionNode(func_name, args)
    
    def parse_array(self) -> ArrayNode:
        """Parse array constant."""
        self.consume_token()  # consume {
        
        rows = []
        current_row = []
        
        while True:
            # Parse element
            current_row.append(self.parse_expression())
            
            token = self.current_token()
            if token and token.type == TokenType.COMMA:
                self.consume_token()  # consume ,
                continue
            elif token and token.type == TokenType.SEMICOLON:
                self.consume_token()  # consume ;
                rows.append(current_row)
                current_row = []
                continue
            elif token and token.type == TokenType.RBRACE:
                self.consume_token()  # consume }
                rows.append(current_row)
                break
            else:
                raise ValueError(f"Expected ',', ';', or '}}' in array constant")
        
        return ArrayNode(rows)

class HybridExcelParser:
    """
    World-class Excel formula parser combining tokenization and AST parsing.
    """
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the parser.
        
        Args:
            context: Optional context with:
                - current_sheet: Current worksheet name
                - named_ranges: Dictionary of named range definitions
                - tables: Dictionary of table definitions
        """
        self.context = context or {}
        self.tokenizer = ExcelTokenizer()
        self.current_sheet = self.context.get('current_sheet', 'Sheet1')
        
        logger.info("✅ HybridExcelParser initialized")
    
    def parse(self, formula: str) -> ASTNode:
        """
        Parse an Excel formula into an AST.
        
        Args:
            formula: Excel formula (with or without =)
            
        Returns:
            Root AST node
        """
        try:
            # Phase 1: Tokenize
            tokens = self.tokenizer.tokenize(formula)
            
            if not tokens:
                raise ValueError("Empty formula")
            
            # Phase 2: Parse
            parser = ExcelParser(tokens, self.tokenizer)
            parser.current_sheet = self.current_sheet
            ast = parser.parse()
            
            # Post-process: Set default sheet for cells without sheet
            self._set_default_sheets(ast)
            
            return ast
            
        except Exception as e:
            logger.error(f"Parse error in formula '{formula}': {e}")
            raise ValueError(f"Invalid Excel formula: {e}")
    
    def _set_default_sheets(self, node: ASTNode):
        """Recursively set default sheet for cells without sheet."""
        if isinstance(node, CellNode) and node.sheet is None:
            node.sheet = self.current_sheet
        elif isinstance(node, RangeNode):
            self._set_default_sheets(node.start)
            self._set_default_sheets(node.end)
        elif isinstance(node, FunctionNode):
            for arg in node.args:
                self._set_default_sheets(arg)
        elif isinstance(node, BinaryOpNode):
            self._set_default_sheets(node.left)
            self._set_default_sheets(node.right)
        elif isinstance(node, UnaryOpNode):
            self._set_default_sheets(node.operand)
        elif isinstance(node, ArrayNode):
            for row in node.elements:
                for elem in row:
                    self._set_default_sheets(elem)

# Testing
if __name__ == '__main__':
    import json
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create parser
    parser = HybridExcelParser({'current_sheet': 'Sheet1'})
    
    # Test formulas
    test_formulas = [
        # Basic arithmetic
        '=1+2*3',
        '=A1+B1',
        '=$A$1+B$2',
        
        # Functions
        '=SUM(A1:A10)',
        '=IF(A1>100,B1*1.1,B1*0.9)',
        '=VLOOKUP(A1,Sheet2!A:D,3,FALSE)',
        
        # Complex formulas
        '=SUM(Revenue)-SUM(Costs)',
        '=IF(AND(A1>0,B1<100),A1*B1,0)',
        
        # Table references
        '=SUM(Table1[Sales])',
        '=SUMIF(Table1[Region],"North",Table1[Sales])',
        
        # Array formulas
        '={1,2,3;4,5,6}',
        
        # Error handling
        '=A1/0',
        '=#DIV/0!',
        
        # More complex
        '=IFERROR(VLOOKUP(A1,Data!$A$1:$C$100,2,FALSE),"Not Found")',
        '=INDEX(Sales,MATCH(A1,Products,0),2)',
    ]
    
    for formula in test_formulas:
        try:
            print(f"\nParsing: {formula}")
            
            # Tokenize first to debug
            tokens = parser.tokenizer.tokenize(formula)
            print(f"Tokens: {[(t.type.name, t.value) for t in tokens]}")
            
            # Parse
            ast = parser.parse(formula)
            print(f"Success! Root node: {type(ast).__name__}")
            
            # Pretty print AST
            def ast_to_dict(node):
                if isinstance(node, NumberNode):
                    return {'type': 'Number', 'value': node.value}
                elif isinstance(node, StringNode):
                    return {'type': 'String', 'value': node.value}
                elif isinstance(node, BooleanNode):
                    return {'type': 'Boolean', 'value': node.value}
                elif isinstance(node, ErrorNode):
                    return {'type': 'Error', 'value': node.error_type}
                elif isinstance(node, CellNode):
                    return {
                        'type': 'Cell',
                        'sheet': node.sheet,
                        'column': node.column,
                        'row': node.row,
                        'absolute_col': node.absolute_col,
                        'absolute_row': node.absolute_row
                    }
                elif isinstance(node, RangeNode):
                    return {
                        'type': 'Range',
                        'start': ast_to_dict(node.start),
                        'end': ast_to_dict(node.end)
                    }
                elif isinstance(node, NamedRangeNode):
                    return {'type': 'NamedRange', 'name': node.name, 'scope': node.scope}
                elif isinstance(node, TableRefNode):
                    return {
                        'type': 'TableRef',
                        'table': node.table_name,
                        'column': node.column_spec,
                        'special': node.special_item
                    }
                elif isinstance(node, FunctionNode):
                    return {
                        'type': 'Function',
                        'name': node.name,
                        'args': [ast_to_dict(arg) for arg in node.args]
                    }
                elif isinstance(node, BinaryOpNode):
                    return {
                        'type': 'BinaryOp',
                        'op': node.op.name,
                        'left': ast_to_dict(node.left),
                        'right': ast_to_dict(node.right)
                    }
                elif isinstance(node, UnaryOpNode):
                    return {
                        'type': 'UnaryOp',
                        'op': node.op.name,
                        'operand': ast_to_dict(node.operand)
                    }
                elif isinstance(node, ArrayNode):
                    return {
                        'type': 'Array',
                        'elements': [[ast_to_dict(elem) for elem in row] for row in node.elements]
                    }
                else:
                    return {'type': 'Unknown', 'class': type(node).__name__}
            
            print(json.dumps(ast_to_dict(ast), indent=2))
            
        except Exception as e:
            print(f"Error: {e}")
