# number of seconds per one sample
NUM_SECS_INPUT = 5
# number of samples per one second
NUM_SAMPLES_SEC = 44100 // 2
# size of signal wave vector
NUM_SAMPLES_INPUT = NUM_SECS_INPUT * NUM_SAMPLES_SEC
# number of audio channel
NUM_AUDIO_CHANNEL = 1
# input shape
INPUT_SHAPE = (NUM_SAMPLES_INPUT, NUM_AUDIO_CHANNEL)