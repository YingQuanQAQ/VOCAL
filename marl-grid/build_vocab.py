import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from vocal_task_specs import TASK_OUTPUTS, TASK_VOCABS, get_supported_tasks


def get_slot_dims(msg_dim, num_slots):
    base = msg_dim // num_slots
    remainder = msg_dim % num_slots
    slot_dims = [base for _ in range(num_slots)]
    for idx in range(remainder):
        slot_dims[idx] += 1
    return slot_dims


def build_slot_embeddings(model, slot_vocab, slot_dim):
    embeddings = model.encode(slot_vocab, convert_to_tensor=True)
    reduced_embeddings = embeddings[:, :slot_dim]
    reduced_embeddings = torch.nn.functional.normalize(
        reduced_embeddings, p=2, dim=1)
    return reduced_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["all", *get_supported_tasks()],
        default="all",
        help="Which task dictionary to build.",
    )
    parser.add_argument(
        "--msg-dim",
        type=int,
        default=10,
        help="Target communication dimension after projection.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/root123/HQ/LanEmerg/all-miniLM-L6-v2",
        help="Local sentence-transformer model path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    print("Loading model from {}...".format(args.model_path))
    model = SentenceTransformer(args.model_path, device="cpu")

    tasks = TASK_VOCABS.keys() if args.task == "all" else [args.task]
    for task in tasks:
        task_spec = TASK_VOCABS[task]
        slot_names = task_spec["slot_names"]
        slot_vocabs = task_spec["slot_vocabs"]
        slot_dims = get_slot_dims(args.msg_dim, len(slot_vocabs))
        slot_embeddings = []

        print("Building offline semantic dictionary for {}...".format(task))
        print(
            "Reducing dimension to msg_dim={} across slots {}...".format(
                args.msg_dim, slot_dims
            )
        )
        for slot_name, slot_vocab, slot_dim in zip(
                slot_names, slot_vocabs, slot_dims):
            print(
                "  slot={} size={} dim={}".format(
                    slot_name, len(slot_vocab), slot_dim
                )
            )
            slot_embeddings.append(
                build_slot_embeddings(model, slot_vocab, slot_dim))

        payload = {
            "task": task,
            "msg_dim": args.msg_dim,
            "slot_names": slot_names,
            "slot_vocabs": slot_vocabs,
            "slot_dims": slot_dims,
            "slot_embeddings": slot_embeddings,
        }

        output_file = repo_root / TASK_OUTPUTS[task]
        torch.save(payload, output_file)
        print(
            "Saved {} embeddings to {} with slot dims {}".format(
                task, output_file, slot_dims
            )
        )


if __name__ == "__main__":
    main()
