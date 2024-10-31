from collections import OrderedDict
import logging
import sys
import torch

def setup_log(log_path):
    log_format = '%(message)s'
    logging.basicConfig(filename=log_path, filemode='w', format=log_format, encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

def save_checkpoint(probe, optimizer, epoch, stats, path):
	# remove embedding model parameters
	probe_state = OrderedDict()
	for param, value in probe.state_dict().items():
		if param.startswith('_emb.'): continue
		probe_state[param] = value

	torch.save({
		'epoch': epoch,
		'stats': stats,
		'probe_state': probe_state,
		'optimizer': optimizer.state_dict()
	}, path)
	logging.info(f"Saved checkpoint to '{path}'.")