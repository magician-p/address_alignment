import json

from config import PROCESSED_DATA_PATH, Dataformat


def get_address(texts, labels):
    results = []
    with open(PROCESSED_DATA_PATH / 'id2label.json', 'r', encoding='utf-8') as f:
        id2label = json.load(f)

    for text, label_sequence in zip(texts, labels):
        characters = [character for character in text if not character.isspace()]
        tags = [id2label.get(str(label), label) for label in label_sequence]
        data = Dataformat()
        index = 0

        while index < min(len(characters), len(tags)):
            prefix, separator, entity_type = tags[index].partition('-')
            field_name = entity_type.capitalize()

            if separator and hasattr(data, field_name) and prefix == 'S':
                setattr(data, field_name, characters[index])
            elif separator and hasattr(data, field_name) and prefix == 'B':
                end = index + 1
                while end < len(tags) and tags[end] == f'I-{entity_type}':
                    end += 1
                if end < len(tags) and tags[end] == f'E-{entity_type}':
                    end += 1
                setattr(data, field_name, ''.join(characters[index:end]))
                index = end - 1

            index += 1

        results.append(data)

    return results