# from lingfeat import extractor

# from readability_transformers.features import Feature

# class LingFeat(Feature):
#     def __init__(self, subgroups):
#         self.subgroups = subgroups
    
#     def extract(self, text: str) -> dict:
#         LingFeat = extractor.pass_text(text)
#         LingFeat.preprocess()
#         features = {}
#         for one_group in self.subgroups:
#             one_group_features = getattr(LingFeat, one_group)()
#             features = {**features, **one_group_features}
#         return features

        
        