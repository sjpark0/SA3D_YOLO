_base_ = './nerf_unbounded_default.py'

expname = 'dcvgo_Set2_unbounded'

data = dict(
    datadir='./data/360_v2/Set2',
    factor=4, # 1558x1039
    movie_render_kwargs=dict(
        shift_y=0.1,
        scale_r=0.9,
        pitch_deg=-20,
    ),
)

