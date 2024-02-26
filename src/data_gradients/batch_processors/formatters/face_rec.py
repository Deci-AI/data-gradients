import face_recognition

import os
import argparse


def find_matching_files(known_dir, unknown_dir, tolerance):

    for known in os.listdir(known_dir):
        known_image = face_recognition.load_image_file(os.path.join(known_dir, known))
        known_encoding = face_recognition.face_encodings(known_image)[0]
        for unknown in os.listdir(unknown_dir):
            unknown_image = face_recognition.load_image_file(os.path.join(unknown_dir, unknown))
            unknown_encoding = face_recognition.face_encodings(unknown_image)
            for unknown_encoding_item in unknown_encoding:
                results = face_recognition.compare_faces([unknown_encoding_item], known_encoding, tolerance=tolerance)
                if results[0]:
                    print(f"{known} and {unknown} match!")


def main():
    parser = argparse.ArgumentParser(description="Compare files in two directories")
    parser.add_argument("known", help="First directory to compare")
    parser.add_argument("unknown", help="Second directory to compare")
    parser.add_argument("--tolerance", "-t", required=False, default=0.6, help="higher is more false alarms.")
    args = parser.parse_args()

    find_matching_files(args.known, args.unknown, args.tolerance)


if __name__ == "__main__":
    main()