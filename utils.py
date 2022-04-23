def convert_to_lower(text):
    return text.lower()


def remove_numbers(text):
    text = re.sub(r'd+', '', text)
    return text


def remove_http(text):
    text = re.sub('https?://t.co/[A-Za-z0-9]*', ' ', text)
    return text


def remove_short_words(text):
    text = re.sub(r'bw{1,2}b', '', text)
    return text


def remove_short_words(text):
    text = re.sub(r'bw{1,2}b', '', text)
    return text


def remove_punctuation(text):
    punctuations = """!()[]{};«№»:'",`./?@=#$-(%^)+&[*_]~"""
    no_punctuation = ''

    for char in text:
        if char not in punctuations:
            no_punctuation = no_punctuation + char
    return no_punctuation


def remove_white_space(text):
    text = text.strip()
    return text


def toknizing(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)

    ## Remove Stopwords from tokens

    result = [i for i in tokens if not i in stop_words]

    return result
