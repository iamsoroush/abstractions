import argparse
from pathlib import Path
from abstractions import Orchestrator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name',
                        type=str,
                        help='name of the folder which contains config file',
                        required=True)

    parser.add_argument('--data_dir',
                        type=str,
                        help='absolute path to directory of the dataset',
                        required=True)

    parser.add_argument('--project_root',
                        type=str,
                        help='absolute path to directory of the project(cloned repository)',
                        required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data_dir = Path(args.data_dir)
    run_name = str(args.run_name)
    project_root = Path(args.project_root)

    orchestrator = Orchestrator(run_name=run_name, data_dir=data_dir, project_root=project_root)
    orchestrator.run()
