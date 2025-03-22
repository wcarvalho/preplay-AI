import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from tqdm.notebook import tqdm

DEFAULT_TITLE_SIZE = 16
DEFAULT_XLABEL_SIZE = 12
DEFAULT_YLABEL_SIZE = 14
DEFAULT_LEGEND_SIZE = 12


default_colors = {
    "reddish purple": (204/255, 121/255, 167/255),
    "yellow": (240/255, 228/255, 66/255),
    "orange": (230/255, 159/255, 0.0),
    "vermillion": (213/255, 94/255, 0.0),
    "sky blue": (86/255, 180/255, 233/255),
    "bluish green": (0.0, 158/255, 115/255),
    "blue": (0.0, 114/255, 178/255),
    "black": "#2f2f2e",
    'dark gray': "#666666",
    'light gray': "#999999",
    'purple': '#CC79A7',
    'nice purple': '#9B80E6',
    "pretty blue": "#679FE5",
    "google blue": "#186CED",
    "google orange": "#FFB700",
    'white': "#FFFFFF",
}

model_colors = {
    'ql': default_colors['dark gray'],
    'ql_sf': default_colors['light gray'],
    'dyna': default_colors['pretty blue'],
    'preplay': default_colors["vermillion"],
}

model_names = {
    'ql': 'Q-learning + 1-step prediction (@10M)',
    'ql_sf': 'Q-learning + successor features (@10M)',
    'dyna': 'Dyna (@1M)',
    'preplay': 'Multi-task preplay (@1M)',
}

model_order = [
    'qlearning',
    'usfa',
    'dyna',
    'preplay',
]

crafter_achievements_names = [
    "Collect Coal",
    "Collect Drink",
    "Collect Iron",
    "Collect Stone",
    "Defeat Skeleton",
    "Defeat Zombie",
    "Eat Cow",
    "Make Stone Pickaxe",
    "Make Stone Sword",
    "Make Wood Pickaxe",
    #"Make Wood Sword",
    #"Place Furnace",
    #"Place Stone",
    #"Place Table",
    "Make Arrow",
    "Collect Drink",
    "Place Torch",
    "Make Torch",
    "Collect Diamond",
    #"Collect Sapling",
    #"Collect Wood",
    #"Eat Plant",
    #"Make Iron Pickaxe",
    #"Make Iron Sword",
    #"Place Plant",
    #"Wake Up",
]
crafter_achievements = [k.lower().replace(" ", "_")
                        for k in crafter_achievements_names]
crafter_achievement_metrics = [f"Achievements/{a}" for a in crafter_achievements]
metrics = ['0.score'] + crafter_achievement_metrics


def get_runs(group, name, entity='wcarvalho92', project='craftax'):
    api = wandb.Api()
    return api.runs(
        f"{entity}/{project}",
        filters={
            'group': group,
            **({"display_name": name} if name else {}),
        })

def get_metric_data_by_group(
    model_to_group=None,
    debug=False):
    """Retrieves raw achievement data from Weights & Biases experiments by group.

    Args:
        setting (str, optional): The metric prefix used in W&B logging.
            Defaults to "evaluator_performance-achievements-64".
        info (dict, optional): Dictionary mapping model names to group names.
            If None, uses default configuration.
            Format: {
                'model_key': 'group_name',
                ...
            }

    Returns:
        pandas.DataFrame: A dataframe containing raw achievement data with columns:
            - model: The model identifier
            - setting: The metric setting used
            - metric: The metric name
            - value: The metric value
            - run_id: The W&B run ID
    """
    # Initialize empty lists to store data
    data = []
    os.makedirs('data', exist_ok=True)

    # Collect data for each model and achievement
    for model, group in tqdm(model_to_group.items(), desc=f"Models", leave=True):
        # Create cache filename
        if debug:
            cache_file = os.path.join(
                'data', f'{model}_{group}_debug_raw.json')
        else:
            cache_file = os.path.join(
                'data', f'{model}_{group}_raw.json')

        # Try to load cached data
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                print(f"Loaded {cache_file}")
                data.extend(json.load(f))
            continue

        model_data = []
        print(group)
        runs = get_runs(group, name=None)

        for run in tqdm(runs, desc=f"Processing {model} runs", leave=True):
            history = run.history()
            keys = sorted(run.summary.keys())
            if len(keys) == 0:
                print(f"No keys found for {run.group}/{run.name}")
                continue
            for key in keys:
              if 'Achievements' in key:
                  parts = key.split('/')
                  setting = parts[0]
                  metric = '/'.join(parts[1:])  # Join remaining parts with '/'
              elif '0.score' in key:
                  setting, metric = key.split('/')
              else:
                  continue
              value = history[key].max()
              model_data.append({
                  'model': model,
                  'setting': setting,
                  'group': group,
                  'name': run.name,
                  'metric': metric,
                  'value': value,
                  'run_id': run.id,
              })
              if debug: break 
            if debug: break

        # Save this model's data to cache
        if model_data:
            with open(cache_file, 'w') as f:
                json.dump(model_data, f)
                print(f"Saved {cache_file}")
            data.extend(model_data)

    return pd.DataFrame(data)

def plot_achievement_bars(df, n=64, fig=None, ax=None, figsize=(15, 5)):
    # Create figure and axis if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Filter data for specific setting
    setting = f"evaluator_performance-achievements-{n}"
    df = df[df['setting'] == setting]

    # Get unique models and achievements
    models = df['model'].unique()
    n_models = len(models)
    n_achievements = len(crafter_achievement_metrics)

    # Set up positions for grouped bars
    bar_width = 0.8 / n_models
    x_pos = np.arange(n_achievements)

    # Sort achievements based on preplay's mean values
    preplay_data = df[df['model'] == 'preplay']
    achievement_means = {metric: preplay_data[preplay_data['metric'] == metric]['value'].mean() 
                        for metric in crafter_achievement_metrics}
    sorted_metrics = sorted(crafter_achievement_metrics, 
                          key=lambda x: achievement_means[x],
                          reverse=True)

    # Plot bars for each model
    bars = {}  # Store bar containers for each model
    for i, model in enumerate(models):
        data = df[df['model'] == model]
        means = []
        sems = []
        for metric in sorted_metrics:
            metric_data = data[data['metric'] == metric]['value']
            means.append(metric_data.mean())
            sems.append(metric_data.sem())
        
        container = ax.bar(x_pos + i * bar_width - (n_models-1)*bar_width/2, 
                          means,
                          bar_width,
                          yerr=sems,
                          label=model_names[model.replace('-', '_')],
                          color=model_colors[model.replace('-', '_')],
                          capsize=3)
        bars[model] = {'means': means, 'sems': sems, 'container': container}

    # Add stars for non-overlapping error bars between preplay and dyna
    for idx, metric in enumerate(sorted_metrics):
        preplay_mean = bars['preplay']['means'][idx]
        preplay_sem = bars['preplay']['sems'][idx]
        dyna_mean = bars['dyna']['means'][idx]
        dyna_sem = bars['dyna']['sems'][idx]

        # Check if error bars don't overlap
        preplay_low = preplay_mean - preplay_sem
        preplay_high = preplay_mean + preplay_sem
        dyna_low = dyna_mean - dyna_sem
        dyna_high = dyna_mean + dyna_sem

        if (preplay_low > dyna_high) or (dyna_low > preplay_high):
            # Place star above the higher bar
            higher_mean = max(preplay_mean, dyna_mean)
            plt.text(idx, higher_mean + max(preplay_sem, dyna_sem), '*', 
                    ha='center', va='bottom', fontsize=14)

    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels([metric.split('/')[-1].replace('_', ' ').title() for metric in sorted_metrics], 
                       rotation=45, ha='right', fontsize=DEFAULT_XLABEL_SIZE)
    ax.set_ylabel('Success Rate', fontsize=DEFAULT_YLABEL_SIZE)
    #ax.set_yscale('log')

    # Set y-axis limit to 110% of max value
    ymax = df['value'].max()
    ymin = df['value'].min()
    ax.set_ylim(ymin, ymax * 1.25)

    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(0.5, .9), loc='center', ncol=len(models), fontsize=DEFAULT_LEGEND_SIZE)
    ax.set_title(f'Per-Achievement Generalization Success Rates given {n} Unique Training Environments', fontsize=DEFAULT_TITLE_SIZE, pad=20)
    
    fig.tight_layout()
    return fig, ax

def plot_training_envs_score(df, ntraining_envs, ax=None, figsize=(5, 5), include_actor=True):
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique models
    models = df['model'].unique()

    # Plot lines and points for each model
    for model in models:
        # Plot evaluator performance (original functionality)
        model_data = []
        model_sems = []
        x_pos = []
        
        for n_env in ntraining_envs:
            setting = f"evaluator_performance-{n_env}"
            data = df[(df['model'] == model) &
                      (df['setting'] == setting) &
                      (df['metric'] == f'0.score')]

            mean = data['value'].mean()
            sem = data['value'].sem()

            if len(data) == 0:
                continue
            model_data.append(mean)
            model_sems.append(sem)
            x_pos.append(n_env)
        
        if len(model_data) > 0:
            # Plot line connecting points
            ax.plot(x_pos, model_data, 
                    '-o',  # Line style with circles for points
                    label=model_names[model.replace('-', '_')] + ' (evaluator)',
                    color=model_colors[model.replace('-', '_')],
                    linewidth=2,
                    markersize=8)
            
            # Add error bars
            ax.errorbar(x_pos, model_data,
                        yerr=model_sems,
                        fmt='none',  # No connecting line
                        color=model_colors[model.replace('-', '_')],
                        capsize=5)
        
        # Plot actor performance if requested
        if include_actor:
            actor_data = []
            actor_sems = []
            actor_x_pos = []
            
            for n_env in ntraining_envs:
                setting = f"actor_performance-{n_env}"
                data = df[(df['model'] == model) &
                        (df['setting'] == setting) &
                        (df['metric'] == f'0.score')]

                mean = data['value'].mean()
                sem = data['value'].sem()

                if len(data) == 0:
                    continue
                actor_data.append(mean)
                actor_sems.append(sem)
                actor_x_pos.append(n_env)
            
            if len(actor_data) > 0:
                # Plot line connecting points with box markers
                ax.plot(actor_x_pos, actor_data, 
                        '-s',  # Line style with squares for points
                        #label=model_names[model.replace('-', '_')] + ' (actor)',
                        color=model_colors[model.replace('-', '_')],
                        linewidth=2,
                        markersize=8,
                        alpha=0.7,  # Slightly transparent to differentiate
                        linestyle='--')  # Dashed line for actor performance
                
                # Add error bars
                ax.errorbar(actor_x_pos, actor_data,
                            yerr=actor_sems,
                            fmt='none',  # No connecting line
                            color=model_colors[model.replace('-', '_')],
                            capsize=5,
                            alpha=0.7)  # Match transparency

    # Customize plot
    ax.set_xlabel('Number of Unique Training Environments', fontsize=DEFAULT_XLABEL_SIZE)
    ax.set_ylabel('% Maximum Score', fontsize=DEFAULT_YLABEL_SIZE)
    
    title = 'Generalization Performance to \n10,000 Unique Environments'
    #if include_actor:
    #    title += '\n(Solid: Evaluator, Dashed: Actor)'
    ax.set_title(title, fontsize=DEFAULT_TITLE_SIZE)

    # Set x-ticks to match ntraining_envs values
    ax.set_xscale('log', base=2)
    ax.set_xticks(ntraining_envs)
    ax.set_xticklabels(ntraining_envs, fontsize=DEFAULT_XLABEL_SIZE)
    ax.set_xlim(min(ntraining_envs)/1.5, max(ntraining_envs)*1.5)

    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if both types of data are shown
    #if include_actor:
    #    ax.legend(fontsize=DEFAULT_LEGEND_SIZE)

    if ax is None:
        plt.tight_layout()
        
    return ax

if __name__ == "__main__":
    # Plot 1: Y-axis is score, X-axis is number of training environments
    ntraining_envs = [8, 16, 32, 64, 128, 256, 512]
    settings = [f"evaluator_performance-achievements-{n}" for n in ntraining_envs]
    model_to_group = {
        'ql': 'ql-eval-5',
        'ql-sf': 'ql-sf-eval-4',
        'dyna': 'dyna-eval-5',
        'preplay': 'preplay-eval-5',
    }

    dfs = []
    for idx, setting in enumerate(settings):
        df = get_metric_data_by_group(
            setting=setting,
            model_to_group=model_to_group,
            debug=True,
        )
        dfs.append(df)
    df = pd.concat(dfs, axis=0)

