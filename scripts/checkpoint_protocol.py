"""Resolve explicit checkpoint splits and reject validation contamination."""


def checkpoint_split(grouped, checkpoint, legacy_validation):
    known = set(grouped)
    if 'training_filenames' not in checkpoint and 'validation_filenames' not in checkpoint:
        if checkpoint.get('split_manifest'):
            raise ValueError('A custom split requires explicit checkpoint provenance')
        return known - set(legacy_validation), set(legacy_validation)
    if 'training_filenames' not in checkpoint or 'validation_filenames' not in checkpoint:
        raise ValueError('Incomplete checkpoint split provenance')
    training_list = checkpoint['training_filenames']
    validation_list = checkpoint['validation_filenames']
    training, validation = set(training_list), set(validation_list)
    if (len(training) != len(training_list) or len(validation) != len(validation_list)
            or not training or not validation or training & validation
            or not (training | validation).issubset(known)):
        raise ValueError('Invalid or contaminated checkpoint split')
    if checkpoint.get('split_manifest') and {n[:8] for n in training} & {n[:8] for n in validation}:
        raise ValueError('Custom date split contains shared training/validation dates')
    return training, validation
