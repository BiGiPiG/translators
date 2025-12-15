"""
Транслятор для языка выражений с присваиванием и инкрементом (++).
Автор: Паршучихин Богдан, группа КМБО-02-23

Грамматика (в БНФ, контекстно-свободная, эквивалентна регулярной):

<Start>                  ::= <ВыражениеПрисваивания>
<ВыражениеПрисваивания>  ::= <ПростоеВыражение> <ОстальныеПрисваивания>
<ОстальныеПрисваивания>  ::= '=' <ВыражениеПрисваивания> | ε
<ПростоеВыражение>       ::= <МножественныйИнкремент> <Операнд> <МножественныйИнкремент>
<МножественныйИнкремент> ::= '++' <МножественныйИнкремент> | ε
<Операнд>                ::= VAR | INT

Терминалы: '++', '=', VAR (имя переменной), INT (целое число)
"""

import sys
import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

# ------------------------------
# 1. Токены и сканер
# ------------------------------

class TokenType(Enum):
    VAR = "VAR"
    INT = "INT"
    PLUS_PLUS = "++"
    ASSIGN = "="
    EOF = "EOF"

@dataclass
class Token:
    type: TokenType
    value: str
    pos: int

class Scanner:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.tokens: List[Token] = []

    def scan(self) -> List[Token]:
        while self.pos < len(self.source):
            char = self.source[self.pos]

            if char.isspace():
                self.pos += 1
                continue

            if self.pos + 1 < len(self.source) and self.source[self.pos:self.pos+2] == "++":
                self.tokens.append(Token(TokenType.PLUS_PLUS, "++", self.pos))
                self.pos += 2
                continue

            if char == '=':
                self.tokens.append(Token(TokenType.ASSIGN, "=", self.pos))
                self.pos += 1
                continue

            if char.isdigit():
                start = self.pos
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    self.pos += 1
                num = self.source[start:self.pos]
                self.tokens.append(Token(TokenType.INT, num, start))
                continue

            if char.isalpha() or char == '_':
                start = self.pos
                while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
                    self.pos += 1
                ident = self.source[start:self.pos]
                self.tokens.append(Token(TokenType.VAR, ident, start))
                continue

            self.error(f"Неизвестный символ '{char}'")

        self.tokens.append(Token(TokenType.EOF, "", len(self.source)))
        return self.tokens

    def error(self, msg: str):
        raise SyntaxError(f"[Сканер] Ошибка на позиции {self.pos}: {msg}")

# ------------------------------
# 2. Узлы AST (для семантики)
# ------------------------------

class ASTNode:
    pass

@dataclass
class Operand(ASTNode):
    token: Token

@dataclass
class SimpleExpression(ASTNode):
    prefix_incs: int
    operand: Operand
    postfix_incs: int

@dataclass
class Assignment(ASTNode):
    left: SimpleExpression
    right: Optional['Assignment']

# ------------------------------
# 3. Парсер и семантический анализатор
# ------------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        self.ast: Optional[Assignment] = None

    def parse(self) -> Assignment:
        self.ast = self.expression_assignment()
        if not self.match(TokenType.EOF):
            self.error("Ожидался конец входа")
        return self.ast

    def expression_assignment(self) -> Assignment:
        left = self.simple_expression()
        if self.match(TokenType.ASSIGN):
            self.consume(TokenType.ASSIGN)
            right = self.expression_assignment()
            return Assignment(left=left, right=right)
        else:
            return Assignment(left=left, right=None)

    def simple_expression(self) -> SimpleExpression:
        prefix = self.multiple_increment()
        operand = self.operand()
        postfix = self.multiple_increment()
        return SimpleExpression(prefix_incs=prefix, operand=operand, postfix_incs=postfix)

    def multiple_increment(self) -> int:
        count = 0
        while self.match(TokenType.PLUS_PLUS):
            self.consume(TokenType.PLUS_PLUS)
            count += 1
        return count

    def operand(self) -> Operand:
        if self.match(TokenType.VAR):
            token = self.consume(TokenType.VAR)
            return Operand(token=token)
        elif self.match(TokenType.INT):
            token = self.consume(TokenType.INT)
            return Operand(token=token)
        else:
            self.error("Ожидался VAR или INT")

    def peek(self) -> Token:
        return self.tokens[self.current]

    def match(self, token_type: TokenType) -> bool:
        if self.current >= len(self.tokens):
            return False
        return self.peek().type == token_type

    def consume(self, token_type: TokenType) -> Token:
        if self.match(token_type):
            token = self.peek()
            self.current += 1
            return token
        else:
            expected = token_type.value if token_type != TokenType.EOF else "конец"
            self.error(f"Ожидался токен '{expected}', найден '{self.peek().type.value}'")

    def error(self, msg: str):
        token = self.peek()
        pos = token.pos if token.type != TokenType.EOF else len("".join(t.value for t in self.tokens))
        raise SyntaxError(f"[Парсер] Ошибка на позиции {pos}: {msg}")

# ------------------------------
# 4. Семантический анализатор
# ------------------------------

class SemanticAnalyzer:
    def __init__(self, ast: Assignment):
        self.ast = ast

    def analyze(self):
        self.check_assignment_targets(self.ast)

    def check_assignment_targets(self, node: Assignment):
        # В любом присваивании слева может быть только VAR (не INT)
        if node.left.operand.token.type == TokenType.INT:
            pos = node.left.operand.token.pos
            self.error(f"Недопустимое присваивание константе на позиции {pos}")

        # Рекурсивно проверяем правую часть, если она — присваивание
        if node.right is not None:
            self.check_assignment_targets(node.right)

    def error(self, msg: str):
        raise ValueError(f"[Семантика] {msg}")

# ------------------------------
# 5. Основная программа
# ------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Транслятор выражений с присваиванием и инкрементом")
    parser.add_argument("input_file", nargs="?", help="Файл с исходным выражением (если не указан — читается stdin)")
    args = parser.parse_args()

    try:
        if args.input_file:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                source = f.read()
        else:
            print("Введите выражение (Ctrl+D/Ctrl+Z для завершения):", file=sys.stderr)
            source = sys.stdin.read()

        scanner = Scanner(source)
        tokens = scanner.scan()

        parser = Parser(tokens)
        ast = parser.parse()

        sem = SemanticAnalyzer(ast)
        sem.analyze()

        print("Успешно: выражение корректно и семантически допустимо.")

    except (SyntaxError, ValueError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Внутренняя ошибка: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()