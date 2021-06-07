import Utils.TrainMethods as TM

TRAIN_VISION_METHOD = "TRAIN_VISION_METHOD"
TRAIN_TEXT_METHOD = "TRAIN_TEXT_METHOD"
TRAIN_TEXT_BINARY_METHOD = "TRAIN_TEXT_BINARY_METHOD"

TRAIN_METHODS = {TRAIN_VISION_METHOD: (TM.train_vision, TM.test_vision),
                 TRAIN_TEXT_METHOD: (TM.train_text, TM.test_text),
                 TRAIN_TEXT_BINARY_METHOD: (TM.train_text_binary, TM.test_text_binary)}
