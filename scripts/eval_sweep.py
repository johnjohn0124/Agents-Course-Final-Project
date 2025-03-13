import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from ilql_eval import evaluate_value_predictions

def eval_data_diet_sweep():
    data_basedir = './checkpoints/twenty-questions/ilql/data_diet_sweep/'
    data_diets = [100, 500, 1000, 5000]
    seeds = [0, 1, 2, 3, 4]

    for data_diet in data_diets:
        print('DATA DIET:', data_diet)
        for seed in seeds:
            print('SEED:', seed)
            train_run_dir = os.path.join(data_basedir, f"diet-{data_diet}", f"seed-{seed}")
            model_path = os.path.join(train_run_dir, 'best_checkpoint')

            config_path = os.path.join(train_run_dir, 'config.yaml')
            val_data_path = './input_data/twenty-questions/eval_transformed.json'
            output_basedir = os.path.join(
                './ilql_results/data-diet-sweep',
                f'diet-{data_diet}',
                f'seed-{seed}'
            )

            evaluate_value_predictions(
                model_path,
                val_data_path,
                output_basedir,
                config_path
            )

def eval_model_size_sweep():
    data_basedir = './checkpoints/twenty-questions/ilql/model-size-sweep/'
    model_names = [
        'openai-community__gpt2-xl',
        'openai-community__gpt2-large',
        'openai-community__gpt2-medium',
        'openai-community__gpt2',
    ]
    seeds = [0, 1, 2, 3, 4]

    for model_name in model_names:
        print('MODEL NAME:', model_name)
        for seed in seeds:
            print('SEED:', seed)
            train_run_dir = os.path.join(data_basedir, f"model-{model_name}", f"seed-{seed}")
            model_path = os.path.join(train_run_dir, 'best_checkpoint')

            config_path = os.path.join(train_run_dir, 'config.yaml')
            val_data_path = './input_data/twenty-questions/eval_transformed.json'
            output_basedir = os.path.join(
                './ilql_results/model-size-sweep',
                f'model-{model_name}',
                f'seed-{seed}'
            )

            evaluate_value_predictions(
                model_path,
                val_data_path,
                output_basedir,
                config_path
            )

if __name__=="__main__":
    # let's sweep through the results from the data diet sweep
    # eval_data_diet_sweep()
    eval_model_size_sweep()