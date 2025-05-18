from recbole.quick_start import run_recbole
import datetime
import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RecBole experiments.")


    parser.add_argument(
        '--config_file',
        type=str,
        default='recbole_baseline_config.yaml', # Default config file if none specified
        help='Path to the RecBole configuration file (e.g., recbole_baseline_config.yaml)'
    )
    args = parser.parse_args()
    config_file_list = [args.config_file]
    init_seed(config['seed'], config['reproducibility'])

    print(f"Starting RecBole run with config: {config_file_list} at: {datetime.datetime.now(), using seed: {config['seed']}}")

    start_time = datetime.datetime.now()
    run_recbole(config_file_list=config_file_list)
    end_time = datetime.datetime.now()

    print(f"Finished RecBole run at: {end_time}")
    print(f"Total duration: {end_time - start_time}")