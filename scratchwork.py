########
# filters for epsisodes
########
filters = [
    lambda e: housemaze_analysis.success(e) < 1,
    lambda e: not e.timesteps.last().any(),
    None,
]
fnames = ['success', 'termination', 'none']
########
# change by manipulation
########
manipulations = [1, 3]
model_settings = [
    dict(maze_name='big_m1_maze3_shortcut', eval=True),
    dict(maze_name='big_m3_maze1', eval=True),
]
fns = [
    functools.partial(went_to_junction, junction=(2, 14)),
    functools.partial(went_to_junction, junction=(17, 17))
]
titles = [
    'Manipulation 1: Reused training path when shortcut introduced',
    'Manipulation 3: Reused training path when two paths exist'
]



for idx, (manipulation, model_setting, title, fn) in enumerate(zip(manipulations, model_settings, titles, fns)):
    model_setting = model_settings[idx]
    #####################
    # Reuse under conditions
    #####################
    model_fn = lambda e: jax.vmap(fn)(e).mean(-1)
    data = dict(
        # vector of people
        human=get_human_data(fn=fn, filter_fn=None, manipulation=manipulation, eval=True),
        human_success=get_human_data(fn=fn, filter_fn=lambda e: housemaze_analysis.success(e) < 1, manipulation=manipulation, eval=True),
        human_failure=get_human_data(fn=fn, filter_fn=lambda e: housemaze_analysis.success(e), manipulation=manipulation, eval=True),
        human_terminate=get_human_data(fn=fn, filter_fn=lambda e: not e.timesteps.last().any(), manipulation=manipulation, eval=True),
        # vector of seeds
        qlearning=get_model_data(fn=fn, algo="qlearning", **model_setting),
        dyna=get_model_data(fn=fn, algo="dynaq_shared", **model_setting),
        bfs=get_model_data(fn=fn, algo='bfs', **model_setting),
        dfs=get_model_data(fn=fn, algo='dfs', **model_setting),
    )
    housemaze_analysis.bar_plot_results(
        data,
        title=title,
        ylabel='Proportion')

    #####################
    # Success rate
    #####################
    fn = housemaze_analysis.success
    data = dict(
        # vector of people
        human=get_human_data(fn=fn, manipulation=manipulation, eval=True),
        # # vector of seeds
        qlearning=get_model_data(fn=fn, algo="qlearning", **model_setting),
        dyna=get_model_data(fn=fn, algo="dynaq_shared", **model_setting),
        bfs=get_model_data(fn=fn, algo='bfs', **model_setting),
        dfs=get_model_data(fn=fn, algo='dfs', **model_setting),
    )
    
    # pprint(reused_path_data)
    housemaze_analysis.bar_plot_results(
        data,
        title=f'Manipulation {manipulation}: Success rate',
        ylabel='Rate')
    #####################
    # Termination rate
    #####################
    fn = lambda e: e.timesteps.last().any()
    data = dict(
        # vector of people
        human=get_human_data(fn=fn, manipulation=manipulation, eval=True),
        # # vector of seeds
        qlearning=get_model_data(fn=fn, algo="qlearning", **model_setting),
        dyna=get_model_data(fn=fn, algo="dynaq_shared", **model_setting),
        bfs=get_model_data(fn=fn, algo='bfs', **model_setting),
        dfs=get_model_data(fn=fn, algo='dfs', **model_setting),
    )
    
    # pprint(reused_path_data)
    housemaze_analysis.bar_plot_results(
        data,
        title=f'Manipulation {manipulation}: Termination rate',
        ylabel='Rate')

    