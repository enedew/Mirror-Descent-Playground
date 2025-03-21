import ast 
import torch

class FunctionParser():
    def __init__(self, input):
        self.input = input


    def string_to_lambda(self):
        # dictionary of allowed names or functions, to be expanded upon
        allowed_funcs = {
            "sin": torch.sin,
            "cos": torch.cos,
            "tan": torch.tan,
            "exp": torch.exp,
            "sqrt": torch.sqrt,
            "log": torch.log,
            "pi": torch.tensor(torch.pi),
        }
        
        # check the string can be parsed as an expression
        try:
            expr_ast = ast.parse(self.input, mode="eval")
        except Exception as parse_err:
            raise ValueError("Invalid expression") from parse_err

        # identify all the components of the expression
        detected_names = set()
        for node in ast.walk(expr_ast):
            if isinstance(node, ast.Name):
                detected_names.add(node.id)
        disallowed_names = ["eval", "exec"]
        # ensure nothing outside of the allowed_names dict is used - this is a safety precausion as I realised 
        # having the ability to inject any lambda function into my code might not be the smartest idea
        # also making sure to filter out disallowed and identify the variables
        variable_names = []
        for name in detected_names: 
            if name in disallowed_names:
                raise ValueError(f"Use of {name} is not allowed")
            elif name not in allowed_funcs: 
                variable_names.append(name)

        code = compile(expr_ast, "<string>", "eval")
        # sorting the variables so theyre applied in a consistent order
        variable_names.sort()
        self.variables = variable_names
        def func(*args): 
            if len(args) != len(variable_names):
                raise ValueError(f"function expects {len(variable_names)} args, but received {len(args)}")
            
            # dict for eval
            dict = {}
            # add allowed functions to dict
            dict.update(allowed_funcs)
            # mapping each variable name to the corresponding argument
            for i, variable in enumerate(variable_names):
                dict[variable] = args[i]

            return eval(code, {"__builtins__": {}}, dict)

        # compile and return the expression
        return func

    # this function just tests the lambda expression with an input value 
    def test_function(self, *args):
        try:
            f = self.string_to_lambda()
            val = f(*args)
            print(f"f{args} = {val}")
        except ValueError as e:
            print(f"Error: {e}")


parser = FunctionParser("X**2 + sin(Y) + 3*X")
parser.test_function(torch.tensor(0.5), torch.tensor(1))
print(torch.tensor(0.5)**2 + torch.sin(torch.tensor(1)) + 3*torch.tensor(0.5))
