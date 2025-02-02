import ast 
import torch

class FunctionParser():
    def __init__(self, input):
        self.input = input


    def string_to_lambda(self):
        # dictionary of allowed names or functions, to be expanded upon
        allowed_names = {
            "x": None,
            "X": None,
            "sin": torch.sin,
            "cos": torch.cos,
            "tan": torch.tan,
            "exp": torch.exp,
            "sqrt": torch.sqrt,
            "log": torch.log,
            "pi": torch.tensor(torch.pi)  
        }
        
        # check the string can be parsed as an expression
        try:
            expr_ast = ast.parse(self.input, mode="eval")
        except Exception as parse_err:
            raise ValueError("Invalid expression") from parse_err

        # ensure nothing outside of the allowed_names dict is used - this is a safety precausion as I realised 
        # having the ability to inject any lambda function into my code might not be the smartest idea
        for node in ast.walk(expr_ast):
            if isinstance(node, ast.Name):
                if node.id not in allowed_names:
                    raise ValueError(f"Usage of name '{node.id}' is not permitted.")

        # compile and return the expression
        code = compile(expr_ast, "<string>", "eval")
        return lambda x: eval(code, {"__builtins__": {}}, dict(allowed_names, x=x, X=x))

    # this function just tests the lambda expression with an input value 
    def test_function(self):
        try:
            func = self.string_to_lambda()
            # For torch, it's typical to pass tensors:
            x = torch.tensor(0.5)
            print(func(x))  # Expected to print a tensor close to 1
        except ValueError as err:
            print(f"Error converting string to function: {err}")


parser = FunctionParser("X**2 + sin(X) + 3*X")
parser.test_function()
print(0.5**2 + torch.sin(torch.tensor(0.5)) + 3*0.5)
