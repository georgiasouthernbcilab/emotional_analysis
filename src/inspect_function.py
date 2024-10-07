import inspect
import ipywidgets as widgets
from IPython.display import display

# Function to create the inspector
def create_inspector(obj):
    # Create the output widget for displaying help
    output_area = widgets.Output()

    # Function to display help when a button is clicked
    def show_help(option):
        with output_area:
            output_area.clear_output()  # Clear previous output
            print(f"Help for {option}:")
            help(getattr(obj, option))  # Display help for the selected method

    # Function to handle the select button click
    def on_select_click(method):
        signature = inspect.signature(method)
        input_widgets = create_param_inputs(signature)
        
        # Add the execute button
        execute_button = widgets.Button(description="Execute", layout=widgets.Layout(width="100px"))
        execute_button.on_click(lambda _: execute_method(method, input_widgets))
        
        # Display the widgets and execute button
        display(widgets.VBox(input_widgets + [execute_button]))

    # Function to create input widgets for method parameters
    def create_param_inputs(signature):
        input_widgets = []
        for param in signature.parameters.values():
            # Create a text input field
            text_input = widgets.Text(
                description=param.name,
                style={'description_width': 'initial'},  # Adjust description width
                layout=widgets.Layout(width='300px'),
                description_tooltip=f"Enter value for {param.name}",
            )
            input_widgets.append(text_input)
        return input_widgets

    # Function to execute the method with given inputs
    def execute_method(method, input_widgets):
        params = [widget.value for widget in input_widgets]
        result = method(*params)
        print(f"Result: {result}")

    # Create a VBox to hold all the widgets
    options_box = widgets.VBox()

    # Loop through the attributes and methods of the passed object
    for i in dir(obj):
        if '_' in i:  # Skip private attributes
            continue
        
        attr = getattr(obj, i)
        if callable(attr):
            # Create buttons for help and select
            help_button = widgets.Button(description='Help', layout=widgets.Layout(width='80px'))
            select_button = widgets.Button(description='Select', layout=widgets.Layout(width='80px'))
            
            help_button.on_click(lambda _, option=i: show_help(option))
            select_button.on_click(lambda _, method=attr: on_select_click(method))
            
            # Create the label for the method
            method_label = widgets.Label(value=f"{i}")
            signature_label = widgets.Label(value=f"Signature: {inspect.signature(attr)}")
            
            # Organize buttons and labels in HBox and VBox
            hbox_buttons = widgets.HBox([select_button, help_button])
            vbox_method = widgets.VBox([method_label, hbox_buttons, signature_label])

            # Add to options box
            options_box.children += (vbox_method,)

    # Display all options and the output area
    display(options_box, output_area)

# Example usage
# Assuming raw is the MNE Raw object already passed
# create_inspector(raw)
