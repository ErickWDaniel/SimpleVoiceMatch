import sounddevice as sd
import soundfile as sf
import librosa
import os


def record_voice(duration):
    fs = 44100  # Sample rate
    recording = sd.rec(int(fs * duration), samplerate=fs, channels=2)
    print("Recording...")
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    sf.write("user_voice.wav", recording, fs)


from sklearn.metrics.pairwise import cosine_similarity


def compare_voice(reference_file, recorded_file):
    reference_mfccs = librosa.feature.mfcc(librosa.load(reference_file, sr=None)[0])
    recorded_mfccs = librosa.feature.mfcc(librosa.load(recorded_file, sr=None)[0])
    similarity = cosine_similarity(reference_mfccs, recorded_mfccs)[0][0]
    if similarity > 0.5:
        print(f'Similarity: {similarity}\nThe voices match')
    else:
        print(f'Similarity: {similarity}\nThe voices do not match')


def main():
    while True:
        print("1. Record Voice")
        print("2. Compare Voice")
        print("3. Exit")
        choice = int(input("Enter your choice: "))
        if choice == 1:
            record_duration = int(input("Enter the duration of the recording in seconds: "))
            record_voice(record_duration)
            save_response = input("Do you want to save the recording? (y/n) ")
            if save_response.lower() == "y":
                filename = "user_voice.wav"
                i = 1
                while os.path.exists(filename):
                    filename = f"user_voice{i}.wav"
                    i += 1
                os.rename("user_voice.wav", filename)
                print(f"Recording saved as {filename}.")
            else:
                os.remove("user_voice.wav")
                print("Recording discarded.")

        elif choice == 2:
            reference_file = input("Enter the name of the reference file: ")
            recorded_file = input("Enter the name of the recorded file: ")
            compare_voice(reference_file, recorded_file)
        elif choice == 3:
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid choice.")


if __name__ == "__main__":
    main()
