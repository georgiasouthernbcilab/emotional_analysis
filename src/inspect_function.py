# inspect_function.py

import inspect
import ipywidgets as widgets
from IPython.display import display

def create_inspector(object_to_inspect):
    # Create a VBox to hold all the widgets
    options_box = widgets.VBox()
    output_area = widgets.Output()  # Create an Output widget to display help

    # Function to display help when a button is clicked
    def show_help(option):
        with output_area:
            # Clear previous output
            output_area.clear_output()  
            print(f"Help for {option}:")
            help(getattr(object_to_inspect, option))  # Display help for the selected method

    # Loop through the attributes and methods of the given object
    for i in dir(object_to_inspect):
        if '_' in i:  # Skip private attributes
            continue
        
        # Create a button for each callable attribute
        attr = getattr(object_to_inspect, i)
        if callable(attr):
            # Create a button with a help function
            help_button = widgets.Button(description='Help')
            help_button.on_click(lambda _, option=i: show_help(option))  # Pass the current option

            # Create a label for the method name
            method_label = widgets.Label(value=f"{i} - Signature: {inspect.signature(attr)}")
            
            # Create a horizontal box to hold the label and button
            hbox = widgets.HBox([method_label, help_button])
            
            # Add the horizontal box to the options box
            options_box.children += (hbox,)

    # Display all the options and the output area
    display(options_box, output_area)
