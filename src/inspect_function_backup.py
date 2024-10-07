# inspect_function.py

import inspect
import ipywidgets as widgets
from IPython.display import display, HTML

def create_inspector(object_to_inspect):
    # Custom CSS for dark theme
    dark_theme_css = """
    .widget-label {
        color: white;
    }
    .widget-button {
        background-color: #444;
        color: white;
        border: 1px solid #666;
    }
    .widget-button:hover {
        background-color: #555;
    }
    .widget-hbox {
        background-color: #333;
        padding: 5px;
        border-radius: 5px;
    }
    .widget-vbox {
        background-color: #222;
        padding: 10px;
        border-radius: 5px;
    }
    .output_area {
        background-color: #222;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    """

    # Inject the CSS into the Jupyter notebook
    display(HTML(f'<style>{dark_theme_css}</style>'))

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

            # Create a horizontal box to hold the method name and help button
            hbox = widgets.HBox([
                widgets.Label(value=f"{i}"),  # Method name
                help_button  # Help button
            ])
            
            # Create a label for the signature
            signature_label = widgets.Label(value=f"Signature: {inspect.signature(attr)}")  # Signature
            
            # Create a vertical box for the method and its signature
            method_vbox = widgets.VBox([hbox, signature_label])
            
            # Add the method VBox to the options box
            options_box.children += (method_vbox,)

    # Display all the options and the output area
    display(options_box, output_area)
