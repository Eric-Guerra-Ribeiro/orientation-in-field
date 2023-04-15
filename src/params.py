import random

class VisionParams:

    # nfeatures: The maximum number of features to retain.
    max_nfeatures=5000
    min_nfeatures=100
    # Scale Factor: Pyramid decimation ratio, greater than 1.
    # scaleFactor==2 means the classical pyramid,
    # where each next level has 4x less pixels than the previous,
    # but such a big scale factor will degrade feature matching scores dramatically.
    # On the other hand, too close to 1 scale factor
    # will mean that to cover certain scale range you will need more pyramid levels
    # and so the speed will suffer.
    max_scaleFactor=2.
    min_scaleFactor=1.01

    max_patchSize = 50
    min_patchSize = 10

    dist_ratio_thres = 0.7

    max_checks=100
    min_checks=1

    max_prob = 0.9999
    min_prob = 0.95

    max_threshold = 10
    min_threshold = 1

    def __init__(self, nfeatures, scaleFactor, patchSize, checks, prob, threshold) -> None:

        assert (self.min_nfeatures <= nfeatures <= self.max_nfeatures)
        assert (self.min_scaleFactor <= scaleFactor <= self.max_scaleFactor)
        assert (self.min_patchSize <= patchSize <= self.max_patchSize)
        assert (self.min_checks <= checks <= self.max_checks)
        assert (self.min_threshold <= threshold <= self.max_threshold)
        

        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.patchSize = patchSize
        self.checks = checks
        # USAC ACURATE Parameters
        self.prob = prob
        self.threshold = threshold

    @classmethod
    def construct_random(cls):
        nfeatures = random.randint(cls.min_nfeatures, cls.max_nfeatures)
        scaleFactor = random.uniform(cls.min_scaleFactor, cls.max_scaleFactor)
        patchSize = random.randint(cls.min_patchSize, cls.max_patchSize)
        checks = random.randint(cls.min_checks, cls.max_checks)
        prob = random.uniform(cls.min_prob, cls.max_prob)
        threshold = random.randint(cls.min_threshold, cls.max_threshold)
        return VisionParams(nfeatures, scaleFactor, patchSize, checks, prob, threshold)


    @classmethod
    def default(cls):
        nfeatures = 500
        scaleFactor = 1.2
        patchSize = 31
        checks = 50
        prob = 0.999
        threshold = 1
        return VisionParams(nfeatures, scaleFactor, patchSize, checks, prob, threshold)

    def __str__(self) -> str:
        return (
              f"nfeatures: {self.nfeatures}\n"
            + f"scaleFactor: {self.scaleFactor}\n"
            + f"pathSize: {self.patchSize}\n"
            + f"checks: {self.checks}\n"
            + f"prob: {self.prob}\n"
            + f"threshold: {self.threshold}"
        )
