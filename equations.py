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

    # Initialize x-values
    x_values = np.array(x_values, dtype=float)
    x_min = min(x_values)
    x_max = max(x_values)

    # Initialize y range
    y_min = 0
    y_max = 0
    for func_name in functions_list:
        y_values = []
        
        for val in x_values:
          y_values.append(func_name.subs(x, val))

        y_values = np.array(y_values, dtype=float)
        
        fig.add_trace(go.Scatter(x = x_values, y = y_values, mode='lines+markers', name = str(func_name).replace('**', '^')) )

        # Update y values:
        y_min = min(y_min, min(y_values))
        y_max = max(y_max, max(y_values))

    
    # Add y-axis
    # fig.add_shape(
    #    type="line",
    #    x0=x_min,
    #    x1=x_max,
    #    y0=0,
    #    y1=0,
    #    line=dict(color="red", width=2)  # You can customize the line color and width
    #    )
    
    # Add x-axis
    # fig.add_shape(
    #    type="line",
    #    x0=0,
    #    x1=0,
    #    y0=y_min,
    #    y1=y_max,
    #    line=dict(color="red", width=2)  # You can customize the line color and width
    #    )
    
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
