import plotly.graph_objects as go
import numpy as np

fig = go.Figure()

import sympy as sp
x = sp.symbols('x')

def compare_functions(parent_function, substitute_function, x_values):

    # Clear figure
    fig.data = []

    # Clear layout
    fig.layout = {}
    
    x_values = np.array(x_values, dtype=float)
    y_values = []
    for val in x_values:
          y_values.append(parent_function.subs(x, val))

    y_values = np.array(y_values, dtype=float)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(parent_function).replace('**', '^')) )

    y_values = []
    for val in x_values:
        y_values.append(substitute_function.subs(x, val))

    y_values = np.array(y_values, dtype=float)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name= str(substitute_function).replace('**', '^')) )

    # Update the layout
    fig.update_layout(autosize = False)

    fig.update_layout(title= {'text' : 'Compare Functions', 
                              'xanchor' : 'center', 
                              'yanchor' : 'top', 
                              'y':0.95,
                              'x':0.5}, 
                              xaxis_title="x", yaxis_title="f(x)")

    # Update the figure:
    fig.show()


def plot_function_list(functions_list, x_values):
    
    # Clear figure
    fig.data = []

    # Clear layout
    fig.layout = {}
    
    x_values = np.array(x_values, dtype=float)
    
    for func_name in functions_list:
        y_values = []
        
        for val in x_values:
          y_values.append(func_name.subs(x, val))

        y_values = np.array(y_values, dtype=float)
        
        fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(func_name).replace('**', '^')) )
    
    # Update the layout
    fig.update_layout(autosize = False)

    fig.update_layout(title= {'text' : 'Function Transformations Plot', 
                              'xanchor' : 'center', 
                              'yanchor' : 'top', 
                              'y':0.95,
                              'x':0.5}, 
                              xaxis_title="x", yaxis_title="f(x)")


    # Update the figure:
    fig.show()
        
        
    
    
def plot_function(parent_function, x_values):

    # Clear figure
    fig.data = []

    # Clear layout
    fig.layout = {}
    
    x_values = np.array(x_values, dtype=float)
    y_values = []
    for val in x_values:
          y_values.append(parent_function.subs(x, val))

    y_values = np.array(y_values, dtype=float)
    fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(parent_function).replace('**', '^')) )

    # Update the layout
    fig.update_layout(autosize = False)

    fig.update_layout(title= {'text' : str(parent_function).replace('**', '^') + ' Plot', 
                              'xanchor' : 'center', 
                              'yanchor' : 'top', 
                              'y':0.95,
                              'x':0.5}, 
                              xaxis_title="x", yaxis_title="f(x)")


    # Update the figure:
    fig.show()
