import sys
# from pathlib import Path
import yaml
from pathlib import Path

class Configer:
    def __init__(self, config_path, mode):
        config = yaml.safe_load(open(config_path, encoding='utf8'))
        
        current_path = Path(config['folder_name'])
        ## Status:
        # self.mode = config['mode']
        self.mode = mode
        
        ## Datasets(Input/Output):
        self.datasets_fold = current_path.joinpath(config['datasets']['datasets_fold'])
        self.update = current_path.joinpath(config['datasets']['datasets_fold'], config['datasets']['update'])
        self.test = current_path.joinpath(config['datasets']['datasets_fold'], config['datasets']['test'])
        self.predict = current_path.joinpath(config['datasets']['datasets_fold'], config['datasets']['predict'])
        self.train_file = current_path.joinpath(config['datasets']['datasets_fold'], config['datasets']['train_file'])
        self.dev_file = current_path.joinpath(config['datasets']['datasets_fold'],config['datasets']['dev_file'])
        self.delimiter = config['datasets']['delimiter']

        ## Model parameters
        self.use_pretrained_embedding = self.str2bool(config['model_parameters']['use_pretrained_embedding'])
        self.token_emb_dir = config['model_parameters']['token_emb_dir']
        self.vocabs_dir = current_path/config['model_parameters']['vocabs_dir']
        self.checkpoints_dir = current_path/config['model_parameters']['checkpoints_dir']
        self.log_dir = current_path/config['model_parameters']['log_dir']
        self.predict_dir = current_path/config['model_parameters']['predict_dir']
        self.sen_dir = current_path/config['model_parameters']['sen_dir']

        ## Labeling Scheme
        self.label_scheme = config['label_setting']['label_scheme']
        self.label_level = int(config['label_setting']['label_level'])
        self.hyphen = config['label_setting']['hyphen']
        self.suffix = config['label_setting']['suffix']
        self.labeling_level = config['label_setting']['labeling_level']
        self.measuring_metrics = config['label_setting']['measuring_metrics']

        ## Modelconfiguration
        self.use_crf = self.str2bool(config['model_config']['use_crf'])
        self.use_self_attention = self.str2bool(config['model_config']['use_self_attention'])
        self.cell_type = config['model_config']['cell_type']
        self.biderectional = self.str2bool(config['model_config']['biderectional'])
        self.encoder_layers = int(config['model_config']['encoder_layers'])
        self.embedding_dim = int(config['model_config']['embedding_dim'])
        self.max_sequence_length = int(config['model_config']['max_sequence_length'])
        self.attention_dim = int(config['model_config']['attention_dim'])
        self.hidden_dim = int(config['model_config']['hidden_dim'])
        self.CUDA_VISIBLE_DEVICES = str(config['model_config']['CUDA_VISIBLE_DEVICES'])
        self.seed = int(config['model_config']['seed'])
        
        ## Training Settings:
        self.is_early_stop = self.str2bool(config['model_config']['is_early_stop'])
        self.patient = int(config['model_config']['patient'])
        self.epoch = int(config['model_config']['epoch'])
        self.batch_size = int(config['model_config']['batch_size'])
        self.dropout = float(config['model_config']['dropout'])
        self.learning_rate = float(config['model_config']['learning_rate'])
        self.optimizer = config['model_config']['optimizer']
        self.checkpoint_name = config['model_config']['checkpoint_name']
        self.checkpoints_max_to_keep = int(config['model_config']['checkpoints_max_to_keep'])
        self.print_per_batch = int(config['model_config']['print_per_batch'])

        ## Testing Settings
        # self.output_test_file = config['test_mode']['output_test_file']
        self.is_output_sentence_entity = self.str2bool(config['test_mode']['is_output_sentence_entity'])
        self.output_sentence_entity_file = config['test_mode']['output_sentence_entity_file']

        ## Api service Settings
        self.ip = config['api_mode']['ip']
        self.port = config['api_mode']['port']

        ## Prediction Settings
        self.predict_file = Path(config['folder_name'])/config['model_parameters']['predict_dir']


    def str2bool(self, string):
        if string == "True" or string == "true" or string == "TRUE":
            return True
        else:
            return False

    def show_data_summary(self, logger):
        logger.info("\n")
        logger.info("++" * 20 + "URATION SUMMARY" + "++" * 20)
        logger.info(" Status:")
        logger.info("     mode               : %s" % (self.mode))
        logger.info(" " + "++" * 20)
        logger.info(" Datasets:")
        logger.info("     datasets       fold: %s" % (self.datasets_fold))
        logger.info("     train          file: %s" % (self.train_file))
        logger.info("     developing     file: %s" % (self.dev_file))
        logger.info("     pre trained embedin: %s" % (self.use_pretrained_embedding))
        logger.info("     embedding      file: %s" % (self.token_emb_dir))
        logger.info("     vocab           dir: %s" % (self.vocabs_dir))
        logger.info("     delimiter          : %s" % (self.delimiter))
        logger.info("     checkpoints     dir: %s" % (self.checkpoints_dir))
        logger.info("     log             dir: %s" % (self.log_dir))
        logger.info("     predict         dir: %s" % (self.predict_dir))
        logger.info("     sen             dir: %s" % (self.sen_dir))
        logger.info(" " + "++" * 20)
        logger.info("Labeling Scheme:")
        logger.info("     label        scheme: %s" % (self.label_scheme))
        logger.info("     label         level: %s" % (self.label_level))
        logger.info("     suffixs           : %s" % (self.suffix))
        logger.info("     labeling_level     : %s" % (self.labeling_level))
        logger.info("     measuring   metrics: %s" % (self.measuring_metrics))
        logger.info(" " + "++" * 20)
        logger.info("Model configuration:")
        logger.info("     use             crf: %s" % (self.use_crf))
        logger.info("     use  self attention: %s" % (self.use_self_attention))
        logger.info("     cell           type: %s" % (self.cell_type))
        logger.info("     biderectional      : %s" % (self.biderectional))
        logger.info("     encoder      layers: %s" % (self.encoder_layers))
        logger.info("     embedding       dim: %s" % (self.embedding_dim))

        logger.info("     max sequence length: %s" % (self.max_sequence_length))
        logger.info("     attention       dim: %s" % (self.attention_dim))
        logger.info("     hidden          dim: %s" % (self.hidden_dim))
        logger.info("     CUDA VISIBLE DEVICE: %s" % (self.CUDA_VISIBLE_DEVICES))
        logger.info("     seed               : %s" % (self.seed))
        logger.info(" " + "++" * 20)
        logger.info(" Training Settings:")
        logger.info("     epoch              : %s" % (self.epoch))
        logger.info("     batch          size: %s" % (self.batch_size))
        logger.info("     dropout            : %s" % (self.dropout))
        logger.info("     learning       rate: %s" % (self.learning_rate))

        logger.info("     optimizer          : %s" % (self.optimizer))
        logger.info("     checkpoint     name: %s" % (self.checkpoint_name))

        logger.info("     max     checkpoints: %s" % (self.checkpoints_max_to_keep))
        logger.info("     print   per   batch: %s" % (self.print_per_batch))

        logger.info("     is   early     stop: %s" % (self.is_early_stop))
        logger.info("     patient            : %s" % (self.patient))
        logger.info(" " + "++" * 20)
        logger.info(" Training Settings:")
        logger.info("     output sent and ent: %s" % (self.is_output_sentence_entity))
        logger.info("     output sen&ent file: %s" % (self.output_sentence_entity_file))
        logger.info(" " + "++" * 20)
        logger.info(" Api service Settings:")
        logger.info("     ip                 : %s" % (self.ip))
        logger.info("     port               : %s" % (self.port))

        logger.info("++" * 20 + " URATION SUMMARY END" + "++" * 20)
        logger.info('\n\n')
        sys.stdout.flush()
