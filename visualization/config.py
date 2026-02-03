"""
Color and style configurations for visualizations
"""

# Color scheme for all visualizations
COLORS = {
    # Points
    'prior_z': '#7f7f7f',        # gray - prior z ~ N(0, I)
    'prior_z_infer': '#1f77b4',   # blue - prior z for inference
    'source_x0': '#7f7f7f',        # gray - prior z ~ N(0, I)
    'source_x0_infer': '#1f77b4',   # blue - prior z for inference
    'data_x': '#ff7f0e',         # orange - data samples x
    'latent_z_hat': '#2ca02c',   # green - encoded latent ẑ
    'output_x_hat': '#d62728',   # red - decoding output D(E(x))
    'infer_x_hat': '#9467bd',    # purple - inference output D(z)
    'trajectory': '#4472C4',     # blue - for ODE trajectories

    # Lines/connections
    'coupling_train': '#7f7f7f', # gray - training coupling (z → x)
    'coupling_infer': '#9467bd', # purple - inference coupling (z → D(z))
    'encode_line': '#ff7f0e',    # orange - encoding (x → ẑ)
    'decode_line': '#d62728',    # red - decoding (ẑ → x̂)
    'infer_line': '#9467bd',     # purple - inference line (z → D(z))
}

# Plot style parameters
PLOT_PARAMS = {
    # Figure size (square for better visualization)
    'figsize': (10, 10),

    # Point sizes & markers
    'point_size': 20,
    'point_alpha': 0.6,
    'marker_data_x': 'x',

    # Line styles
    'line_alpha': 0.3,
    'line_width': 0.5,

    # Epsilon noise for x-axis separation (very small)
    'epsilon_scale': 0.005,

    # Subplot layout
    'left_x': 0,      # x-position for left panel (source/prior)
    'right_x': 1,     # x-position for right panel (target/data)

    # X-axis shifts for different element types
    'shift_prior_z': -0.008,
    'shift_latent_z_hat': 0.008,
    'shift_data_x': -0.008,
    'shift_output_x_hat': 0.0,
    'shift_infer_x_hat': 0.008,
    'shift_z_train': -0.006,
    'shift_z_infer': 0.006,

    # Axis limits padding
    'y_padding': 0.5,
}

# Labels
LABELS = {
    'prior_z': 'Prior z ~ N(0,I)',
    'source_z': 'Source z (\eps)',
    'data_x': 'Data x',
    'latent_z_hat': 'Encoded ẑ',
    'output_x_hat': 'Decoding D(E(x))',
    'infer_x_hat': 'x_1',
    'x_0': 'x_0',
    'coupling_train': 'Training coupling',
    'coupling_infer': 'Inference mapping',
    'encode_line': 'Encode (x → ẑ)',
    'decode_line': 'Decode (ẑ → x̂)',
    'infer_line': 'Inference (z → D(z))',
    'trajectory': 'ODE Trajectory',
}


def add_epsilon_noise(values, scale=None):
    """
    Add small random noise to x-axis for better point separation
    """
    import numpy as np
    if scale is None:
        scale = PLOT_PARAMS['epsilon_scale']
    return np.random.uniform(-scale, scale, len(values))


def setup_plot_style():
    """
    Set up matplotlib plot style
    """
    import matplotlib.pyplot as plt
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = PLOT_PARAMS['figsize']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.linewidth'] = PLOT_PARAMS['line_width']
