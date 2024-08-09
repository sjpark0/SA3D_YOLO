_base_ = './nerf_unbounded_default.py'

expname = 'dcvgo_dome_unbounded'

data = dict(
    datadir='./data/360_v2/dome',
    factor=4, # 1558x1039
    movie_render_kwargs=dict(
        shift_y=-0.5,
        scale_r=0.5,
        pitch_deg=-0,
    ),
)

