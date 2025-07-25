import plotly.graph_objects as go
import numpy as np
import matplotlib.colors as mcolors
from scipy.__config__ import show



#def color_name_to_rgb(color_name):
    #return mcolors.to_rgb(color_name)
def color_name_to_rgb(color_name):
    """
    Convert a color name to an RGB tuple.
    Uses an extended color list if the color name is not recognized by Matplotlib.
    
    Parameters:
    - color_name: The color name (e.g., 'red', 'teal', 'turquoise').
    
    Returns:
    - A tuple representing the RGB color.
    """
    # Extended color list with hex codes
    extended_colors = {
    "teal": "#008080",        # Pair 1
    "coral": "#FF7F50",
    "navy": "#000080",        # Pair 2
    "gold": "#FFD700",
    "crimson": "#DC143C",     # Pair 3
    "turquoise": "#40E0D0",
    "mint": "#98FF98",        # Pair 4
    "salmon": "#FA8072",
    "lavender": "#E6E6FA",    # Pair 5
    "olive": "#808000",
    "lime": "#00FF00",        # Pair 6
    "maroon": "#800000",
    "indigo": "#4B0082",      # Pair 7
    "orange": "#FFA500",
    "cyan": "#00FFFF",        # Pair 8
    "magenta": "#FF00FF",
    "violet": "#EE82EE",      # Pair 9
    "brown": "#A52A2A",
    "royalblue": "#4169E1",   # Pair 10
    "yellow": "#FFFF00",
    "pink": "#FFC0CB",        # Pair 11
    "darkgreen": "#006400",
    "skyblue": "#87CEEB",     # Pair 12
    "chocolate": "#D2691E",
    "plum": "#DDA0DD",        # Pair 13
    "forestgreen": "#228B22",
    "slategray": "#708090",   # Pair 14
    "lightcoral": "#F08080",
    "darkorchid": "#9932CC",  # Pair 15
    "chartreuse": "#7FFF00",
    # Add more colors as needed
    }

    # Check in the extended color list first
    if color_name.lower() in extended_colors:
        hex_color = extended_colors[color_name.lower()]
        return mcolors.hex2color(hex_color)
    
    # If not found in extended list, use Matplotlib's named colors
    try:
        return mcolors.to_rgb(color_name)
    except ValueError:
        raise ValueError(f"Color '{color_name}' is not recognized. Use a valid color name or hex code.")


def viz1pcl(v1_array, color="red", marker_size=3, opacity=1, legend_label='Point Cloud', show_legend=True,save_figure=False, save_path='test.pdf', file_format='pdf'):
    """
    Visualize a 3D point cloud using Plotly with a transparent sphere surface.
    
    Parameters:
    - v1_array: A (n, 3) array of 3D points.
    - color: Color of the point cloud.
    - marker_size: Size of the markers.
    - opacity: Opacity of the markers.
    - legend_label: Label for the point cloud in the legend.
    """
    if v1_array.shape[1] != 3:
        raise ValueError("v1_array must have shape (n, 3)")

    # Create the scatter plot for v1_array
    trace = go.Scatter3d(
        x=v1_array[:, 0],
        y=v1_array[:, 1],
        z=v1_array[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color=color, opacity=opacity),
        name=legend_label
    )

    data = [trace]

    # Generate a transparent sphere surface
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    sphere_surface = go.Surface(
        x=x, 
        y=y, 
        z=z, 
        opacity=0.03,  # Set a low opacity for the sphere
        colorscale=[[0, 'blue'], [1, 'blue']],
        showscale=False,
        name='Sphere Surface'
    )

    data.append(sphere_surface)

    # Define Layout
    layout = go.Layout(
        title="3D Point Cloud Visualization",
        scene=dict(
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            zaxis=dict(visible=False, showticklabels=False),
            aspectmode="cube",  # Equal aspect ratio
            bgcolor="white"  # Set background color to white
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Tight Layout
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.95,  # x-position in the lower right corner
            y=0.05,  # y-position in the lower right corner
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.8)",  # Slightly transparent white background
            font=dict(size=16),  
            showlegend=show_legend
        )
    )

    # Create figure and add trace
    fig = go.Figure(data=data, layout=layout)

    # Additional updates to layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            zaxis=dict(visible=False, showticklabels=False),
            camera=dict(eye=dict(x=0.9, y=-1.1, z=0.4))  # Set camera position
        )
    )
    
    # Add the reference frame with X, Y, Z axes
    fig.add_trace(go.Scatter3d(
        x=[0, 0.25, 0, 0, 0, 0],
        y=[0, 0, 0, 0.25, 0, 0],
        z=[0, 0, 0, 0, 0, 0.25],
        mode='lines',
        line=dict(width=6, color='red'),
        name='X Axis',
        showlegend=False
    ))



    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0.25],
        z=[0, 0],
        mode='lines',
        line=dict(width=6, color='green'),
        name='Y Axis',
        showlegend=False
    ))



    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 0.25],
        mode='lines',
        line=dict(width=6, color='blue'),
        name='Z Axis',
        showlegend=False
    ))



    # Origin point
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(size=8, color='black'),
        name='Origin',
        showlegend=False
    ))
    
    if save_figure:
        if save_path is not None:
            if file_format == 'pdf':
                fig.write_image(save_path)

    fig.show()

def viz2pcl(v1_array, v2_array=None, color1="coral", color2="royalblue", marker_size=3, opacity=1,show_legend=True, legend_1='Destination', legend_2='Source_gt', save_figure=False, save_path='test.pdf', file_format='pdf'):
    """
    Visualize one or two 3D point clouds using Plotly with a transparent sphere surface.
    
    Parameters:
    - v1_array: A (n, 3) array of 3D points.
    - v2_array: An optional (m, 3) array of 3D points for the second point cloud.
    - color1: Color of the first point cloud.
    - color2: Color of the second point cloud.
    - marker_size: Size of the markers.
    - opacity: Opacity of the markers.
    - legend_1: Label for the first point cloud in the legend.
    - legend_2: Label for the second point cloud in the legend.
    """
    if v1_array.shape[1] != 3:
        raise ValueError("v1_array must have shape (n, 3)")
    
    # Create the scatter plot for v1_array
    trace1 = go.Scatter3d(
        x=v1_array[:, 0],
        y=v1_array[:, 1],
        z=v1_array[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color=color1, opacity=opacity),
        name=legend_1,
        showlegend=show_legend
    )

    data = [trace1]

    # Create the scatter plot for v2_array if provided
    if v2_array is not None:
        if v2_array.shape[1] != 3:
            raise ValueError("v2_array must have shape (m, 3)")
        trace2 = go.Scatter3d(
            x=v2_array[:, 0],
            y=v2_array[:, 1],
            z=v2_array[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=color2, opacity=opacity),
            name=legend_2,
            showlegend=show_legend
        )
        data.append(trace2)

    # Generate a transparent sphere surface
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    sphere_surface = go.Surface(
        x=x, 
        y=y, 
        z=z, 
        opacity=0.03,  # Set a low opacity for the sphere
        colorscale=[[0, 'blue'], [1, 'blue']],
        showscale=False,
        name='Sphere Surface'
    )

    data.append(sphere_surface)

    # Define Layout
    layout = go.Layout(
        title="3D Point Cloud Visualization",
        scene=dict(
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            zaxis=dict(visible=False, showticklabels=False),
            aspectmode="cube",  # Equal aspect ratio
            bgcolor="white"  # Set background color to white
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Tight Layout
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.95,  # x-position in the lower right corner
            y=0.05,  # y-position in the lower right corner
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.8)",  # Slightly transparent white background
            font=dict(size=16)  
        )
    )

    # Create figure and add trace
    fig = go.Figure(data=data, layout=layout)

    # Additional updates to layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            zaxis=dict(visible=False, showticklabels=False),
            camera=dict(eye=dict(x=0.9, y=-1.1, z=0.4))  # Set camera position
        )
    )
    
    # Add the reference frame with X, Y, Z axes
    fig.add_trace(go.Scatter3d(
        x=[0, 0.25, 0, 0, 0, 0],
        y=[0, 0, 0, 0.25, 0, 0],
        z=[0, 0, 0, 0, 0, 0.25],
        mode='lines',
        line=dict(width=6, color='red'),
        name='X Axis',
        showlegend=False
    ))



    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0.25],
        z=[0, 0],
        mode='lines',
        line=dict(width=6, color='green'),
        name='Y Axis',
        showlegend=False
    ))



    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 0.25],
        mode='lines',
        line=dict(width=6, color='blue'),
        name='Z Axis',
        showlegend=False
    ))



    # Origin point
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(size=8, color='black'),
        name='Origin',
        showlegend=False
    ))
        
    if save_figure:
        if save_path is not None:
            if file_format == 'pdf':
                fig.write_image(save_path)

    fig.show()

def viz3pcl(v1_array, v2_array=None, v3_array=None, color1="coral", color2="royalblue", color3="teal",show_legend=True, marker_size=3, opacity=1, legend_1='Destination', legend_2='Source_gt', legend_3='Source_aligned',save_figure=False,save_path='test.pdf',file_format='pdf'):
    """
    Visualize up to three 3D point clouds using Plotly with a transparent sphere surface.
    
    Parameters:
    - v1_array: A (n, 3) array of 3D points.
    - v2_array: An optional (m, 3) array of 3D points for the second point cloud.
    - v3_array: An optional (o, 3) array of 3D points for the third point cloud.
    - color1: Color of the first point cloud.
    - color2: Color of the second point cloud.
    - color3: Color of the third point cloud.
    - marker_size: Size of the markers.
    - opacity: Opacity of the markers.
    - legend_1: Label for the first point cloud in the legend.
    - legend_2: Label for the second point cloud in the legend.
    - legend_3: Label for the third point cloud in the legend.
    """
    if v1_array.shape[1] != 3:
        raise ValueError("v1_array must have shape (n, 3)")
    
    # Create the scatter plot for v1_array
    trace1 = go.Scatter3d(
        x=v1_array[:, 0],
        y=v1_array[:, 1],
        z=v1_array[:, 2],
        mode="markers",
        marker=dict(size=marker_size, color=color1, opacity=opacity),
        name=legend_1,
        showlegend=show_legend
    )

    data = [trace1]

    # Create the scatter plot for v2_array if provided
    if v2_array is not None:
        if v2_array.shape[1] != 3:
            raise ValueError("v2_array must have shape (m, 3)")
        trace2 = go.Scatter3d(
            x=v2_array[:, 0],
            y=v2_array[:, 1],
            z=v2_array[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=color2, opacity=opacity),
            name=legend_2,
            showlegend=show_legend
        )
        data.append(trace2)

    # Create the scatter plot for v3_array if provided
    if v3_array is not None:
        if v3_array.shape[1] != 3:
            raise ValueError("v3_array must have shape (o, 3)")
        trace3 = go.Scatter3d(
            x=v3_array[:, 0],
            y=v3_array[:, 1],
            z=v3_array[:, 2],
            mode="markers",
            marker=dict(size=marker_size, color=color3, opacity=opacity),
            name=legend_3,
            showlegend=show_legend
        )
        data.append(trace3)

    # Generate a transparent sphere surface
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    sphere_surface = go.Surface(
        x=x, 
        y=y, 
        z=z, 
        opacity=0.03,  # Set a low opacity for the sphere
        colorscale=[[0, 'blue'], [1, 'blue']],
        showscale=False,
        name='Sphere Surface'
    )

    data.append(sphere_surface)

    # Define Layout
    layout = go.Layout(
        title="3D Point Cloud Visualization",
        scene=dict(
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            zaxis=dict(visible=False, showticklabels=False),
            aspectmode="cube",  # Equal aspect ratio
            bgcolor="white"  # Set background color to white
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Tight Layout
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            x=0.95,  # x-position in the lower right corner
            y=0.05,  # y-position in the lower right corner
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255, 255, 255, 0.8)",  # Slightly transparent white background
            font=dict(size=16)  # Font size for the legend
        )
    )

    # Create figure and add trace
    fig = go.Figure(data=data, layout=layout)

    # Additional updates to layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            zaxis=dict(visible=False, showticklabels=False),
            camera=dict(eye=dict(x=0.9, y=-1.1, z=0.4))  # Set camera position
        )
    )
    
    # Add the reference frame with X, Y, Z axes
    fig.add_trace(go.Scatter3d(
        x=[0, 0.25, 0, 0, 0, 0],
        y=[0, 0, 0, 0.25, 0, 0],
        z=[0, 0, 0, 0, 0, 0.25],
        mode='lines',
        line=dict(width=6, color='red'),
        name='X Axis',
        showlegend=False
    ))



    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0.25],
        z=[0, 0],
        mode='lines',
        line=dict(width=6, color='green'),
        name='Y Axis',
        showlegend=False
    ))



    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 0.25],
        mode='lines',
        line=dict(width=6, color='blue'),
        name='Z Axis',
        showlegend=False
    ))



    # Origin point
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(size=8, color='black'),
        name='Origin',
        showlegend=False
    ))
    
    if save_figure:
        if save_path is not None:
            if file_format == 'pdf':
                fig.write_image(save_path)
    
    fig.show()
