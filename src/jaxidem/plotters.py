#!/usr/bin/env python3

"""
Various functions to use plotly to plot objects from jaxidem
"""

import jax
import jax.numpy as jnp

from jaxtyping import ArrayLike, PyTree
from typing import Callable, NamedTuple

from jaxidem.utils import st_data
from jaxidem.idem import Kernel

from jaxidem.utils import create_grid

import imageio
import os


try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    from plotly.subplots import make_subplots
except ImportError as e:
    raise ImportError(
        "Plotting requires the 'plotting' extra. "
        "Install it with: pip install jaxidem[plotting]"
    ) from e



# Catppuccin Mocha palette
cat = {
    "base": "#181825",
    "mantle": "#181825",
    "crust": "#11111b",
    "text": "#cdd6f4",
    "subtext1": "#bac2de",
    "surface0": "#313244",
    "surface1": "#45475a",
    "overlay0": "#6c7086",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "sky": "#89dceb",
    "teal": "#94e2d5",
    "green": "#a6e3a1",
    "yellow": "#f9e2af",
    "peach": "#fab387",
    "maroon": "#eba0ac",
    "red": "#f38ba8",
    "mauve": "#cba6f7",
    "pink": "#f5c2e7",
    "flamingo": "#f2cdcd",
    "rosewater": "#f5e0dc",
}

def apply_catppuccin_mocha(fig, fontsize=20):
    
    # applies catppuccin mocha theme to a figure
    fig.update_layout(
        paper_bgcolor=cat["base"],
        plot_bgcolor=cat["mantle"],
        font=dict(color=cat["text"], size=fontsize),
        title_font=dict(color=cat["text"]),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=cat["surface0"],
        zeroline=False, linecolor=cat["overlay0"], ticks="outside",
        tickfont=dict(color=cat["subtext1"])
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=cat["surface0"],
        zeroline=False, linecolor=cat["overlay0"], ticks="outside",
        tickfont=dict(color=cat["subtext1"])
    )
    fig.update_layout(
        annotations=[
            ann.update(font=dict(size=fontsize)) or ann
            for ann in fig.layout.annotations
        ]
    )


# creates frames of gridded data
# TODO: add checks to ensure data is sufficiently gridded
def plotly_st_grid(st,
                   title="Spatio-Temporal Field",
                   colorscale="Viridis",
                   sort_axes=True,      # sort unique x, y, t; set False to keep first-seen order
                   show_colorbar=True,
                   ):

    # adapted from (and probably stolen by) an llm. Probably rethink or find a proper attributation.
    
    x = st.x
    y = st.y
    z = st.z
    t = st.t

    #def unique_axis(vals):
    #    if sort_axes:
    #        uniq = jnp.unique(vals)
    #        to_idx = lambda v: jnp.searchsorted(uniq, v)
    #    else:
    #        _, idx = jnp.unique(vals, return_index=True)
    #        order = jnp.sort(idx)
    #        uniq = vals[order]
    #        mapping = {v: i for i, v in enumerate(uniq.tolist())}
    #        to_idx = lambda v: jnp.fromiter((mapping[vi] for vi in v.tolist()), dtype=int)
    #    return uniq, to_idx

    #ux, xi = unique_axis(x)
    #uy, yi = unique_axis(y)
    #ut, ti = unique_axis(t)

    ux = jnp.unique(st.x)
    uy = jnp.unique(st.y)
    ut = st.full_times
    
    #if callable(xi):
    #    xi = xi(x)
    #    yi = yi(y)
    #    ti = ti(t)

    nx, ny = len(ux), len(uy)

    idx = jnp.searchsorted(ux, x, side="left")
    idx_clipped = jnp.clip(idx, 0, ux.size - 1)
    matches = ux[idx_clipped] == x
    xi = jnp.where(matches, idx_clipped, -1)

    idy = jnp.searchsorted(uy, y, side="left")
    idy_clipped = jnp.clip(idy, 0, uy.size - 1)
    matches = uy[idy_clipped] == y
    yi = jnp.where(matches, idy_clipped, -1)

    ti = st.t.astype("int32")
    
    z_grid = jnp.full((st.T-1, ny, nx), jnp.nan, dtype=float)
    z_grid = z_grid.at[ti, yi, xi].set(z)

    zmin = float(jnp.nanmin(z_grid))
    zmax = float(jnp.nanmax(z_grid))
    
    frames = []
    for i in range(st.T-1):
        frames.append(
            go.Frame(
                name=f"t = {ut[i]}",
                data=[
                    go.Heatmap(
                        z=z_grid[i],
                        x=ux,
                        y=uy,
                        colorscale=colorscale,
                        zmin=zmin,
                        zmax=zmax,
                        colorbar=dict(title="") if show_colorbar else None,
                        showscale=show_colorbar,
                    )
                ],
            )
        )

    return frames


# creates frames of scattered data
def plotly_st_scatter(st,
                      title="Spatio-Temporal Field",
                      colorscale="Viridis",
                      ):

    T = st.T

    klass = go.Scattergl

    zmin = float(jnp.nanmin(st.z))
    zmax = float(jnp.nanmax(st.z))

    frames = []
    for t in range(T):
        x = st.coords_tree[t][:,0]
        y = st.coords_tree[t][:,1]
        z = st.zs_tree[t]

        trace = go.Scattergl(
            x=x.tolist(),
            y=y.tolist(),
            mode="markers",
            marker=dict(
                size = 15,
                color=z.tolist(),
                colorscale=colorscale,
                cmin=zmin,
                cmax=zmax,
                showscale=True
            ),
            showlegend=False
        )
        frames.append(go.Frame(name=str(t), data=[trace]))
        
    coloraxis = dict(
        colorscale=colorscale,
        cmin=float(zmin),
        cmax=float(zmax),
        colorbar=dict(title="z")  # customize as needed
    )
        
    return frames


def make_st_fig(frames, ncols=3, spacing = [0.02, 0.02]):
    
    """
    Combines a list of plotly frames into a grid plot.
    """
    
    nframes = len(frames)
    nrows = int(jnp.ceil(nframes / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f"t = {int(ti)+1}" for ti in range(nframes)],
        horizontal_spacing=spacing[0],
        vertical_spacing=spacing[1]
    )

    for idx, fr in enumerate(frames):
        row = int(jnp.floor(idx /ncols)) + 1
        col = int(idx % ncols) + 1
        fig.add_trace(fr.data[0], row=row, col=col)

    for i in range(1, nrows * ncols + 1):
        fig.update_xaxes(matches='x', row=(i-1)//ncols + 1, col=(i-1)%ncols + 1)
        fig.update_yaxes(matches='y', row=(i-1)//ncols + 1, col=(i-1)%ncols + 1)
            
    # Also anchor y to x for square cells
    for c in range(1, 4):
        fig.update_yaxes(scaleanchor=f"x{c}", row=1, col=c)
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0)

    return fig


def make_interactive_fig(frames):
    
    fig = go.Figure(
        data=frames[0].data,  # initial frame
        frames=frames         # all frames
    )
    
    fig.update_layout(
        sliders=[{
            "active": 0,
            "pad": {"t": 50},
            "steps": [{
                "label": f"{i}",
                "method": "animate",
                "args": [[f.name], {"mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0}}]
            } for i, f in enumerate(frames)]
        }]
    )

    return fig


# NEED TO MAKE IT REMOVE THE TEMP FOLDER
def save_st_gif(frames, filename, apply_theme=apply_catppuccin_mocha):

    # Create a temporary folder
    os.makedirs("frames", exist_ok=True)

    # Save each frame as PNG
    for i, frame in enumerate(frames):
        fig = go.Figure(data=frame.data)
        apply_theme(fig, fontsize=20)
        fig.write_image(f"frames/frame_{i:03d}.png", width=800, height=600)

    # Stitch into GIF
    images = [imageio.imread(f"frames/frame_{i:03d}.png") for i in range(len(frames))]
    imageio.mimsave(filename, images, duration=0.1, loop=0)  # duration in seconds per frame


def kernel_plot(kernel):

    """
    Half-finished; how do i determine the bounds?
    """
    
    @jax.vmap
    def offset(s):
        return -jnp.array(
            [
                kernel.params[2] @ kernel.basis[2].vfun(s),
                kernel.params[3] @ kernel.basis[3].vfun(s),
            ]
        )

    bounds = jnp.array([[0, 1], [0, 1]])
    grid = create_grid(bounds, jnp.array([10, 10])).coords
    x, y = grid[:, 0], grid[:, 1]
    offsets = offset(grid)
    dx, dy = offsets[:, 0], offsets[:, 1]

    fig = ff.create_quiver(x, y, dx*5, dy*5)

    fig.update_layout(
        xaxis=dict(scaleanchor="y", showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    return fig

    
