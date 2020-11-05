import os, tqdm, librosa, numpy as np

WAV_DIR = "dataset/wav"
NPY_DIR = "dataset/npy"

if __name__ == "__main__":
    meta_data = open("dataset/metadata.txt", "w+")

    for voice_dir in tqdm.tqdm(os.listdir(WAV_DIR)):
        npy_data = np.array([])
        for wav_file_name in os.listdir("%s/%s" % (WAV_DIR, voice_dir)):
            wav, sr = librosa.load("%s/%s/%s" % (WAV_DIR, voice_dir, wav_file_name))
            npy_data = np.concatenate((npy_data, wav))

        os.makedirs("%s/%s" % (NPY_DIR, voice_dir), exist_ok=True)
        np.save("%s/%s/data.npy" % (NPY_DIR, voice_dir), npy_data)

        print(
            "%s\t%s" % (voice_dir, npy_data.shape[0] / 22050),
            file=meta_data,
        )