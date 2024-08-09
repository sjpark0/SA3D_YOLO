_base_ = './nerf_unbounded_default.py'

expname = 'dcvgo_MIV_unbounded'

data = dict(
    datadir='./data/360_v2/MIV',
    factor=4, # 1558x1039
    movie_render_kwargs=dict(
        shift_y=-0.5,
        scale_r=0.9,
        pitch_deg=-20,
    ),
)

