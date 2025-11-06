import copy
import os
from datetime import datetime
import yaml
import shutil

from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain

from train_deim_v2.DEIMv2.engine.core import YAMLConfig
from train_deim_v2.DEIMv2.engine.misc import dist_utils
from train_deim_v2.DEIMv2.engine.solver import TASKS
from train_deim_v2.utils.ikutils import prepare_dataset
from train_deim_v2.utils.load_model import resolve_config_and_weights, MODEL_MAPPING



# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainDeimV2Param(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        dataset_folder = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "dataset")
        self.cfg["dataset_folder"] = dataset_folder
        self.cfg["model_name"] = "n_coco"
        self.cfg["model_weight_file"] = ""
        self.cfg["epochs"] = 50
        self.cfg["batch_size"] = 8
        self.cfg["input_size"] = 640
        self.cfg["dataset_split_ratio"] = 0.9
        self.cfg["workers"] = 0
        self.cfg["weight_decay"] = 0.0001
        self.cfg["lr"] = 0.0005
        self.cfg["config_file"] = ""
        self.cfg["output_folder"] = os.path.dirname(
            os.path.realpath(__file__)) + "/runs/"

    def set_values(self, params):
        self.cfg["dataset_folder"] = str(params["dataset_folder"])
        self.cfg["model_name"] = str(params["model_name"])
        self.cfg["model_weight_file"] = str(params["model_weight_file"])
        self.cfg["epochs"] = int(params["epochs"])
        self.cfg["batch_size"] = int(params["batch_size"])
        self.cfg["input_size"] = int(params["input_size"])
        self.cfg["workers"] = int(params["workers"])
        self.cfg["weight_decay"] = float(params["weight_decay"])
        self.cfg["lr"] = float(params["lr"])
        self.cfg["config_file"] = str(params["config_file"])
        self.cfg["dataset_split_ratio"] = float(
            params["dataset_split_ratio"])
        self.cfg["output_folder"] = str(params["output_folder"])


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainDeimV2(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Create parameters object
        if param is None:
            self.set_param_object(TrainDeimV2Param())
        else:
            self.set_param_object(copy.deepcopy(param))

        # self.enable_mlflow(True)
        self.experiment_name = None
        self.cfg_folder = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "DEIMv2", "configs")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def set_output_dir(self, param):
        # Create output folder
        model_name = param.cfg["model_name"]
        self.experiment_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(param.cfg["output_folder"], exist_ok=True)
        output_folder = os.path.join(
            param.cfg["output_folder"], self.experiment_name)
        os.makedirs(output_folder, exist_ok=True)

        return output_folder

    def load_config(self, cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.unsafe_load(f)

    def _find_config_file(self, model_name):
        if not model_name:
            raise ValueError("Model name must be provided to resolve configuration file.")

        if os.path.isabs(model_name) and os.path.isfile(model_name):
            return model_name

        # Check if model_name is in mapping and get config filename
        if model_name in MODEL_MAPPING:
            _, config_filename, _ = MODEL_MAPPING[model_name]
            base_dir = os.path.dirname(os.path.realpath(__file__))
            search_roots = [
                self.cfg_folder,
                base_dir,
                os.path.join(base_dir, "DEIMv2"),
                os.path.join(self.cfg_folder, "deimv2"),
                os.path.join(self.cfg_folder, "deim_dfine"),
                os.path.join(self.cfg_folder, "deim_rtdetrv2")
            ]

            for root in search_roots:
                candidate = os.path.join(root, config_filename)
                if os.path.isfile(candidate):
                    return candidate

        # Fallback to original behavior
        base_dir = os.path.dirname(os.path.realpath(__file__))
        potential_names = [model_name]
        if not model_name.endswith(('.yml', '.yaml')):
            potential_names.append(f"{model_name}.yml")

        search_roots = [self.cfg_folder,
                        base_dir,
                        os.path.join(base_dir, "DEIMv2"),
                        os.path.join(self.cfg_folder, "deimv2"),
                        os.path.join(self.cfg_folder, "deim_dfine"),
                        os.path.join(self.cfg_folder, "deim_rtdetrv2")]

        for name in potential_names:
            for root in search_roots:
                candidate = os.path.join(root, name)
                if os.path.isfile(candidate):
                    return candidate

        raise FileNotFoundError(
            f"Unable to locate configuration file for model '{model_name}'."
        )

    @staticmethod
    def _update_dataloader_cfg(dataloader_cfg, batch_size, workers):
        if not isinstance(dataloader_cfg, dict):
            return

        if 'total_batch_size' in dataloader_cfg:
            dataloader_cfg['total_batch_size'] = batch_size
        if 'batch_size' in dataloader_cfg:
            dataloader_cfg['batch_size'] = batch_size
        dataloader_cfg['num_workers'] = workers

    def _apply_param_overrides(self, cfg, param, dataset_info, output_folder):
        cfg.yaml_cfg['output_dir'] = output_folder
        cfg.yaml_cfg['num_classes'] = dataset_info['nc']
        cfg.yaml_cfg['remap_mscoco_category'] = False

        train_dataset_cfg = cfg.yaml_cfg.get('train_dataloader', {}).get('dataset', {})
        val_dataset_cfg = cfg.yaml_cfg.get('val_dataloader', {}).get('dataset', {})

        train_dataset_cfg['img_folder'] = dataset_info['train_img_dir']
        train_dataset_cfg['ann_file'] = dataset_info['train_annot_file']
        val_dataset_cfg['img_folder'] = dataset_info['val_img_dir']
        val_dataset_cfg['ann_file'] = dataset_info['val_annot_file']

        batch_size = max(1, int(param.cfg['batch_size']))
        workers = max(0, int(param.cfg['workers']))

        self._update_dataloader_cfg(cfg.yaml_cfg.get('train_dataloader'), batch_size, workers)
        self._update_dataloader_cfg(cfg.yaml_cfg.get('val_dataloader'), batch_size, workers)

        optimizer_cfg = cfg.yaml_cfg.get('optimizer', {})
        if isinstance(optimizer_cfg, dict):
            optimizer_cfg['lr'] = float(param.cfg['lr'])
            optimizer_cfg['weight_decay'] = float(param.cfg['weight_decay'])

        size = int(param.cfg['input_size'])
        if size % 32 != 0:
            adjusted = (size // 32) * 32
            print(f"Updating input size from {size} to {adjusted} to be a multiple of 32")
            size = max(32, adjusted)
        size_pair = [size, size]

        cfg.yaml_cfg['eval_spatial_size'] = size_pair

        train_transforms = train_dataset_cfg.get(
            'transforms', {}).get('ops', []) if isinstance(train_dataset_cfg, dict) else []
        for op in train_transforms:
            if isinstance(op, dict):
                if op.get('type') == 'Resize' and 'size' in op:
                    op['size'] = size_pair
                if op.get('type') == 'Mosaic' and 'output_size' in op:
                    op['output_size'] = max(1, size // 2)

        val_transforms = val_dataset_cfg.get(
            'transforms', {}).get('ops', []) if isinstance(val_dataset_cfg, dict) else []
        for op in val_transforms:
            if isinstance(op, dict) and op.get('type') == 'Resize' and 'size' in op:
                op['size'] = size_pair

        collate_cfg = cfg.yaml_cfg.get('train_dataloader', {}).get('collate_fn', {})
        if isinstance(collate_cfg, dict) and 'base_size' in collate_cfg:
            collate_cfg['base_size'] = size

    def _save_training_artifacts(self, cfg, output_folder, dataset_info):
        if not self.experiment_name:
            return

        # copy configs folder to output folder self.cfg_folder
        shutil.copytree(self.cfg_folder, os.path.join(output_folder, 'configs'))

        training_config = os.path.join(
            output_folder, f'config_{self.experiment_name}.yaml')
        # Build a serializable configuration with extra metadata fields
        base_cfg = cfg.yaml_cfg if hasattr(cfg, 'yaml_cfg') else cfg
        try:
            cfg_to_save = copy.deepcopy(base_cfg)
        except Exception:
            cfg_to_save = dict(base_cfg)

        # Add requested fields
        cfg_to_save['num_classes_finetuned'] = dataset_info.get('nc')
        cfg_to_save['class_names'] = dataset_info.get('names')
        cfg_to_save['model_name'] = self.get_param_object().cfg.get('model_name')
        cfg_to_save['__include__'] = [
            'configs/dataset/coco_detection.yml',
            'configs/runtime.yml',
            'configs/base/dataloader.yml',
            'configs/base/optimizer.yml',
            'configs/base/deimv2.yml',
        ]

        with open(training_config, 'w', encoding='utf-8') as file:
            yaml.safe_dump(cfg_to_save, file, sort_keys=False, allow_unicode=True)

        class_names_path = os.path.join(output_folder, 'class_names.txt')
        with open(class_names_path, 'w', encoding='utf-8') as file:
            for name in dataset_info['names']:
                file.write(f"{name}\n")

    def run(self):
        self.begin_task_run()

        try:
            param = self.get_param_object()
            dataset_input = self.get_input(0)

            dataset_info = prepare_dataset(dataset_input.data,
                                           param.cfg['dataset_folder'],
                                           param.cfg['dataset_split_ratio'])
            print(f"\nFinal dataset info: {dataset_info}")

            dist_utils.setup_distributed(print_rank=0, print_method='builtin')

            custom_config = param.cfg.get('config_file') or None
            custom_weight = param.cfg.get('model_weight_file') or None

            _, config_path, resolved_weight = resolve_config_and_weights(
                param.cfg.get('model_name'),
                config_override=custom_config,
                weight_override=custom_weight,
                weights_dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights')
            )

            if config_path is None:
                config_path = self._find_config_file(param.cfg.get('model_name'))

            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            cfg = None
            if custom_config:
                print(f"Using custom configuration file: {config_path}")
                cfg = self.load_config(config_path)
            else:
                print(f"Using configuration file: {config_path}")
                if resolved_weight:
                    print(f"Using pretrained weights: {resolved_weight}")

                output_folder = self.set_output_dir(param)

                cfg_kwargs = {
                    'output_dir': output_folder,
                    'epoches': param.cfg['epochs']
                }
                if resolved_weight:
                    cfg_kwargs['tuning'] = resolved_weight

                cfg = YAMLConfig(config_path, **cfg_kwargs)
                self._apply_param_overrides(cfg, param, dataset_info, output_folder)
                self._save_training_artifacts(cfg, output_folder, dataset_info)

            if hasattr(cfg, 'yaml_cfg'):
                if 'HGNetv2' in cfg.yaml_cfg:
                    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
                if 'DINOv3STAs' in cfg.yaml_cfg and param.cfg.get('model_weight_file'):
                    cfg.yaml_cfg['DINOv3STAs']['weights_path'] = param.cfg['model_weight_file']

            solver = TASKS[cfg.yaml_cfg['task']](cfg)

            if param.cfg.get('test_only', False):
                solver.val()
            else:
                solver.fit()

        finally:
            dist_utils.cleanup()
            self.emit_step_progress()
            self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainDeimV2Factory(dataprocess.CTaskFactory):  # type: ignore[attr-defined]

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)  # type: ignore[attr-defined]
        # Set algorithm information/metadata here
        self.info.name = "train_deim_v2"
        self.info.short_description = "Train DEIMv2: Real-Time Object Detection Meets DINOv3"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Huang, Shihua and Hou, Yongjie and Liu, Longfei and Yu, " \
                            "Xuanlong and Shen, Xi"
        self.info.article = "Real-Time Object Detection Meets DINOv3"
        self.info.journal = "arXiv:2509.20787v2"
        self.info.year = 2025
        self.info.license = "Apache 2.0"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.15.0"

        # Python compatibility
        self.info.min_python_version = "3.9.0"

        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2509.20787"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/train_deim_v2"
        self.info.original_repository = "https://github.com/Intellindust-AI-Lab/DEIMv2"

        # Keywords used for search
        self.info.keywords = "Object Detection,DINOv3,DEIMv2,COCO"
        self.info.algo_type = core.AlgoType.TRAIN
        self.info.algo_tasks = "OBJECT_DETECTION"

        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = True
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create algorithm object
        return TrainDeimV2(self.info.name, param)
