import plotly.graph_objects as go
import numpy as np
import sympy as sp

class graph_funs:
  """
  The class supports function manipulations and plots.
  """

  def __init__(self):
    """ Setup the plot.
    """
    # Create an empty function list
    self.functions_list  = []
    self.functions_names = []

    # Create an empty list of x-values:
    self.domains = []

    # Create an empty list of vertical asymptotes:
    self.vertical_asymptotes = []
    self.vertical_asymptotes_names = []

  def add_vert_asymptotes(self, x_vals, asymp_names=None):
    """ adds vertical list of asymptotes based on x_values
    """
    self.vertical_asymptotes.extend(x_vals)
    if asymp_names is None:
      names = []
      for x in x_vals:
        new_str = "x = "+str(x) 
        names.append(new_str)
      self.vertical_asymptotes_names.extend(names)
    else:
      self.vertical_asymptotes_names.extend(asymp_names)

  def add_fun(self, f, domain, f_name=None):
    """ adds a function to the current list.
    """
    self.functions_list.append(f)
    self.domains.append(domain)
    if f_name is None:
      self.functions_names.append(str(f))
    else:
      self.functions_names.append(f)

  def add_funs(self, fun_list, domain_list, fun_names_list=None):
    """ adds a list of functions to the current list.
    """
    self.functions_list.extend(fun_list)
    self.domains.extend(domain_list)
    if fun_names_list is None:
      self.functions_names.extend(str(fun_list))
    else:
      self.functions_names.extend(fun_names_list)

  def add_vert_translations(self, f, domain, trans_range):
    """ adds vertical translations of f to list of functions.
    """
    x_vals = np.linspace(*trans_range)
    for x_val in x_vals:
      new_fun = f + x_val
      self.functions_list.append(new_fun)
      self.functions_names.append(str(new_fun))
      self.domains.append(domain)
    return 

  def add_hor_translations(self, f, domain, trans_range):
    """ adds horizontal translations of f to list of functions. """
    x = sp.symbols('x')
    min_x = domain[0]
    max_x = domain[1]
    num_of_points = domain[2]

    for x_val in np.linspace(*trans_range):
      new_fun = f.subs(x, x-x_val)
      self.functions_list.append(new_fun)
      self.functions_names.append(str(new_fun))

      new_domain = [min_x+x_val, max_x+x_val, num_of_points]
      self.domains.append(new_domain)
    return 

  def add_refl_across_x(self, f, domain, f_name=None):
    """ adds a reflection of f across x to the functions list. """
    new_fun = -f
    self.functions_list.append(new_fun) 
    self.domains.append(domain)
    if f_name is None:
      self.functions_names.append(str(new_fun))
    else:
      self.functions_names.append(f_name)
    return

  def add_refl_across_y(self, f, domain, f_name=None):
    """ adds a reflection of f across y to the functions list.
    """
    x = sp.symbols('x')
    new_fun = f.subs(x, -x)

    min_x = domain[0]
    max_x = domain[1]
    num_of_points = domain[2]
    new_domain = [-max_x, -min_x, num_of_points]
    self.domains.append(new_domain)

    self.functions_list.append(new_fun) 
    if f_name is None:
      self.functions_names.append(str(new_fun))
    else:
      self.functions_names.append(f_name)
    return

  def add_vert_dilations(self, exp_fun, domain, a_range):
    """ adds vertical dilations of exponential functions to functions list. 
        This is done by multiplying by a.
    """
    for a_val in np.linspace(*a_range):
      new_fun = a_val * exp_fun 
      self.functions_list.append(new_fun)
      self.functions_names.append(str(new_fun))
      self.domains.append(domain)
    return 

  def add_hor_dilations(self, f, domain, x_scale_range):
    """ adds horizontal dilations to functions list.
    """ 
    x = sp.symbols('x')
    min_x = domain[0]
    max_x = domain[1]
    num_of_points = domain[2]

    for x_scale in np.linspace(*x_scale_range):
      new_fun = f.subs(x, x_scale*x)
      self.functions_list.append(new_fun)
      self.functions_names.append(str(new_fun))

      if (x_scale > 0):
        new_domain = [x_scale*min_x, x_scale*max_x, num_of_points]
      else:
        new_domain = [x_scale*max_x, x_scale*min_x, num_of_points]
      self.domains.append(new_domain)
    return new_fun 

  def plot_funs(self, x_axis_name="x", y_axis_name="y", plot_title="Transformations"):
    """ plots a list of functions.
    """
    # Create the figure:
    fig = go.Figure()

    # Clear figure
    fig.data = []

    # Clear layout
    fig.layout = {}

    # Initialize y range
    y_min = 0
    y_max = 0
    x = sp.symbols('x')
    for func_name, str_name, domain in zip(self.functions_list, self.functions_names, self.domains):
        # Generate the y-values
        y_values = []
        x_values = np.linspace(*domain) # Unpack the list of elements.
        for val in x_values:
          y_values.append(func_name.subs(x, val))
        y_values = np.array(y_values, dtype=float)
        
        fig.add_trace(go.Scatter(x = x_values, 
                                      y = y_values, 
                                      mode='lines+markers', 
                                      name = str_name.replace('**', '^'),
                                      showlegend=True) )

        # Update y values:
        y_min = min(y_min, min(y_values))
        y_max = max(y_max, max(y_values))

    # Add all of the asymptotes:
    for x, str_name in zip(self.vertical_asymptotes, self.vertical_asymptotes_names):
      fig.add_shape(
        type="line", x0=x, x1=x,
        y0=y_min, y1=y_max,
        line=dict(color="red", width=2)  # You can customize the line color and width
        )
      
    # Add y-axis
    # fig.add_shape(
    #    type="line",
    #    x0=x_min,
    #    x1=x_max,
    #    y0=0,
    #    y1=0,
    #    line=dict(color="red", width=2)  # You can customize the line color and width
    #    )
    
    
    
    # Update the layout
    fig.update_layout(autosize = False)

    fig.update_layout(title= {'text' : plot_title, 
                              'xanchor' : 'center', 
                              'yanchor' : 'top', 
                              'y':0.95,
                              'x':0.5}, 
                              xaxis_title=x_axis_name, yaxis_title=y_axis_name)


    # Update the figure:
    fig.show()
        

  def __repr__(self) -> str:
    str_rep =  "graph_funs class parameters.\n"  
    str_rep += "Domains: \n"
    str_rep += str(self.domains)+"\n"
    str_rep += "Function names: \n"
    str_rep += str(self.functions_names)+"\n"
    str_rep += "Vertical asymptotes:\n"
    str_rep += str(self.vertical_asymptotes_names)+"\n"
    return(str_rep)
  
  

