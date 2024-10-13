import re
import numpy as np
import pandas as pd

def unicode(text):
    # Define the replacement mappings
    replacements = {
        "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
        "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
        "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
        "Ủy": "Uỷ"
    }
    
    # Define a function to apply the replacements to a single text
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
    
def dupplicate_punctuation(text, pos=[], spans=[]):
    if not pos:
        pos = list(range(len(text)))

    def replace(text, pattern, replacement, pos):
        matches = [0]  # Initialize a list to track the positions of matches.

        # Nested function to handle each regex match.
        def capture_and_replace(match, ret):
            matches.extend([match.start() + 1, match.end()])  # Store the start+1 and end positions of the match.
            return ret  # Return the replacement text for the match.

        # Get the length of the original text.
        l = len(text)

        # Use `re.sub` to find all occurrences of the pattern and replace them.
        # `capture_and_replace` is used as a callback to record match positions.
        text = re.sub(pattern, lambda match: capture_and_replace(match, replacement), text, flags=re.IGNORECASE)

        # Add the length of the modified text to the matches.
        matches.append(l)

        # Split the matches list into pairs of start and end positions.
        slices = np.array_split(matches, int(len(matches) / 2))

        # Adjust the `pos` list according to the changes made in the text.
        res = []
        for s in slices:
            res += pos[s[0]:s[1]]  # Extend `res` with the corresponding slice of `pos`.

        # Ensure the length of the updated `text` matches the updated `pos` list.
        assert len(text) == len(res)

        return text, res  # Return the updated text and the adjusted `pos` list.

    # Collapse duplicated punctuations 
    punc = ',. !?\"\''

    # Perform the replacement for each punctuation character.
    for c in punc:
        pat = f'([{c}]+)'
        text, pos = replace(text, pat, c, pos)

    # Adjust spans according to the new positions.
    new_spans = []
    for start, end in spans:
        new_start = pos.index(start) if start in pos else -1
        new_end = pos.index(end - 1) + 1 if (end - 1) in pos else -1
        if new_start != -1 and new_end != -1:
            new_spans.append([new_start, new_end])

    # Ensure that the length of `text` matches the updated `pos`.
    assert len(text) == len(pos)
    
    return text, pos, new_spans

def deleteIcon(text):
    text = text.lower()
    s = ''
    pattern = r"[a-zA-ZaăâbcdđeêghiklmnoôơpqrstuưvxyàằầbcdđèềghìklmnòồờpqrstùừvxỳáắấbcdđéếghíklmnóốớpqrstúứvxýảẳẩbcdđẻểghỉklmnỏổởpqrstủửvxỷạặậbcdđẹệghịklmnọộợpqrstụựvxỵãẵẫbcdđẽễghĩklmnõỗỡpqrstũữvxỹAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXYÀẰẦBCDĐÈỀGHÌKLMNÒỒỜPQRSTÙỪVXỲÁẮẤBCDĐÉẾGHÍKLMNÓỐỚPQRSTÚỨVXÝẠẶẬBCDĐẸỆGHỊKLMNỌỘỢPQRSTỤỰVXỴẢẲẨBCDĐẺỂGHỈKLMNỎỔỞPQRSTỦỬVXỶÃẴẪBCDĐẼỄGHĨKLMNÕỖỠPQRSTŨỮVXỸ,._]"
    for char in text:
        if char !=' ':
            if len(re.findall(pattern, char)) != 0:
                s+=char
            elif char == '_':
                s+=char
        else:
            s+=char
    s = re.sub('\\s+',' ',s)
    return s.strip()

def to_category_vector(label, classes):
    vector = np.zeros(len(classes)).astype(np.float64)
    index = classes.index(label)
    vector[index] = 1.0
    return vector



def sentence_based_chunking(row):
    # Split the text into sentences using basic punctuation marks.
    text = row['Text']
    tag = row['Tag']
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunked_data = []
    start = 0

    for sentence in sentences:
        # Calculate the end position of the sentence in the original text.
        end = start + len(sentence)
        
        # Append the chunked sentence with its spans and tag.
        chunked_data.append({
            'chunk': sentence,
            'Tag': tag,
        })

        # Update start position for the next sentence.
        start = end + 1  # +1 to account for space after splitting.

    return chunked_data


def data_chunking(data):
    
    formated_data = []


    for _, row in data.iterrows():
        annotations = sentence_based_chunking(row)
        annotations = [annotation.values() for annotation in annotations]
        annotations = [list(annotation) for annotation in annotations]

        # Combine tokens and their corresponding annotations
        formated_data.extend(annotations)
        formated_data.append((None, None))  # Append a marker for sentence separation

    # Create a DataFrame from the formatted data
    df_final = pd.DataFrame(formated_data, columns=['Chunk', 'Tag'])

    # Generate sentence IDs
    sentence_id = []
    sentence = 0
    for chunk in df_final['Chunk']:
        if chunk is not None:
            sentence_id.append(sentence)
        else:
            sentence_id.append(np.nan)
            sentence += 1

    df_final['sentence_id'] = sentence_id
    df_final.dropna(subset=['Chunk'], inplace=True)  # Remove rows where 'Chunk' is None
    df_final['sentence_id'] = df_final['sentence_id'].astype('int64')  # Convert to int64
    df_final = df_final[['Chunk', 'Tag', 'sentence_id']]

    return df_final