def process_and_update(args):
    batch, split, batch_processor, grouped_feature_extractors = args
    for sample in batch_processor.process(batch, split=split):
        for feature_extractors in grouped_feature_extractors.values():
            for feature_extractor in feature_extractors:
                feature_extractor.update(sample)
    yield from batch_processor.process(batch, split=split)
